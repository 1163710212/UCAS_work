import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from prepare_data import load_data
from model import BertRegressionModel
import argparse


#### 定义评价函数
class Critic(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.relu = nn.ReLU()

    def forward(self, predicts, labels):
        predicts = self.relu(predicts)
        l1 = (torch.abs(torch.log((1 + torch.round(predicts)) / (1 + labels))) - 1).sum()
        l2 = -(torch.round(predicts) == labels).sum()
        l = (l1 * 0.7 + l2 * 0.3) / predicts.shape[0]
        return l


#### 定义验证函数
def dev(model, dev_loader, device):
    # 将模型放到服务器上
    model.to(device)
    # 设定模式为验证模式
    model.eval()
    # 设定不会有梯度的改变仅作验证
    loss = Critic(device)
    total_loss = 0
    total_mse_loss = 0
    mse = nn.L1Loss()
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(dev_loader),
                                                                              desc='Dev Itreation:'):
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)
            out_put = model(input_ids, token_type_ids, attention_mask).view(-1)
            predict = out_put.data
            total_loss += loss(predict, labels) * labels.size(0)
            total_mse_loss += mse(predict, labels) * labels.size(0)
            predict = torch.round(predict).type(torch.int)
            labels = torch.round(labels).type(torch.int)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        l = total_loss / total
        mse_loss = total_mse_loss / total
        return res, l, mse_loss


####  定义训练函数
def train(model, train_loader, dev_loader, device, args):
    # 将model放到服务器上
    model.to(device)
    # 设定模型的模式为训练模式
    model.train()
    # 定义模型的损失函数
    criterion = nn.L1Loss()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 学习率的设置
    optimizer_params = {'lr': 1e-5, 'eps': 1e-6}
    # 使用AdamW 主流优化器
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
    # 学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  min_lr=1e-7, patience=5, verbose=True,
                                  threshold=0.0001, eps=1e-08)
    # 记录训练日志
    logger = logging.getLogger("Train")
    logging.basicConfig(level=logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args['log_path'],
                                       mode='a',
                                       encoding='utf-8')
    cons_fmt = logging.Formatter('%(message)s')
    file_fmt = logging.Formatter('%(asctime)s : %(message)s')
    console_handler.setFormatter(cons_fmt)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # 设定训练轮次
    total_epochs = 40
    bestAcc = 0
    best_loss = torch.tensor([100])
    logger.info('Training and verification begin!')
    for epoch in range(total_epochs):
        train_loss = 0
        step_loss = 0
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
            # 从实例化的DataLoader中取出数据，并通过 .to(device)将数据部署到服务器上
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), \
                                                                token_type_ids.to(device), \
                                                                attention_mask.to(device), \
                                                                labels.to(device)
            model.train()
            # 梯度清零
            optimizer.zero_grad()
            # 将数据输入到模型中获得输出
            out_put = model(input_ids, token_type_ids, attention_mask).view(-1)
            # 计算损失
            loss = criterion(out_put, labels)
            predict = out_put.data
            predict = torch.round(predict).type(torch.int)
            labels = torch.round(labels).type(torch.int)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            train_loss += loss.item()
            step_loss += loss.item()
            loss.backward()
            optimizer.step()
            # 每100次进行一次打印
            if (step + 1) % 20 == 0:
                train_acc = correct / total
                logger.info("Train Epoch[{}/{}],step[{}/{}],"
                            "tra_acc{:.6f} %,,step_loss:{:.6f},ep_loss:{:.6f}".format(epoch + 1, total_epochs,
                                                                                      step + 1, len(train_loader),
                                                                                      train_acc * 100, step_loss / 50,
                                                                                      train_loss / step))
                step_loss = 0
            # 每3500次进行一次验证
            if (step + 1) % 100 == 0:
                train_acc = correct / total
                # 调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
                acc, test_loss, test_mse_loss = dev(model, dev_loader, device)
                if bestAcc < acc:
                    bestAcc = acc
                if best_loss > test_loss:
                    best_loss = test_loss
                    # 模型保存路径
                    fold = args['fold']
                    path = f'../user_data/model/bert_fold{fold}_{int(epoch / 5)}.pkl'
                    torch.save(model, path)
                logger.info(
                    "DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %,test_my_loss:{:.6f}, test_mae_loss:{:.6f},best_loss{:.6}".format(
                        epoch + 1, total_epochs, step + 1, len(train_loader),
                        train_acc * 100, bestAcc * 100, acc * 100,
                        test_loss.item(), test_mse_loss.item(), best_loss.item()))
        scheduler.step(bestAcc)


# 1.设置参数
# 定义命令行解析器对象
parser = argparse.ArgumentParser(description='argparse')
# 添加命令行参数
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
# 从命令行中结构化解析参数
args = parser.parse_args()
fold = args.fold
cuda = args.cuda
args = {
    'fold': fold,
    'train_data_path': f'../user_data/data/fold/train_f{fold}.csv',
    'dev_data_path': f'../user_data/data/fold/dev_f{fold}.csv',
    'cuda': f'cuda:{cuda}',
    'batch_size': 4,
    'load_model_path': f'../user_data/model/fold/bert_fold{fold}_0.pkl',
    # 'save_model_path': './model/fold/bert_fold3.pkl',
    'log_path': f'../user_data/log/fold{fold}.text',
}

#### 2.实例化DataLoader
# 设定batch_size
batch_size = args['batch_size']
# 引入数据路径
train_data_path = args['train_data_path']
dev_data_path = args['dev_data_path']
# 调用load_data函数，将数据加载为Tensor形式
train_data = load_data(train_data_path)
dev_data = load_data(dev_data_path)
# 将训练数据和测试数据进行DataLoader实例化
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)

##### 3.实例化模型并进行训练与验证 
device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
# 实例化模型
model = BertRegressionModel(2, device)
#path = args['load_model_path']
#model = torch.load(path)
# 调用训练函数进行训练与验证
train(model, train_loader, dev_loader, device, args)
