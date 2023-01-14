import os
import time

import torch
import numpy as np
from sklearn.metrics import accuracy_score
# from sklearn.metrics import ndcg_score
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import datetime
import sklearn

# print(sklearn.__version__)
# exit


# 将DataFrame划分为训练集和验证集
def train_val_split(dataset, frac=0.8, random_state=None):
    train_set = dataset.sample(frac=frac, random_state=random_state, axis=0)
    val_set = dataset[~dataset.index.isin(train_set.index)]
    return train_set.reset_index(), val_set.reset_index()


class TextSet(Dataset):

    def __init__(self, dataset_df):
        self.dataset = dataset_df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> dict:
        query = self.dataset.loc[item, 'query']
        passage = self.dataset.loc[item, 'passage']
        is_relevant = self.dataset.loc[item, 'is_relevant']
        return {'text': query + '[SEP]' + passage, 'label': is_relevant}


class Trainer:
    MAX_TEXT_LENGTH = 226
    BATCH_SIZE = 4

    def __init__(self, model, tokenizer, dataset, frac=0.8):
        self.model = model
        train_set, val_set = train_val_split(dataset, frac, random_state=None)
        self.train_loader = DataLoader(
            TextSet(train_set), batch_size=self.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(
            TextSet(val_set), batch_size=self.BATCH_SIZE, shuffle=True)
        self.tokenizer = tokenizer
        self.writer = SummaryWriter('runs/scalar_example')  # tensorboard

    # 训练并验证模型
    def train(self, device, epoch_num):
        # 初始化训练参数
        model = self.model
        train_loader = self.train_loader
        optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                                    num_training_steps=epoch_num*len(train_loader))
        criterion = torch.nn.BCELoss()  # 损失函数

        if not os.path.isdir('../models'):  # 存模型directory
            os.makedirs('../models')

        # 开始训练
        self.model.train()
        print('********* Training Start *********\n')
        best_acc = 0
        # best_ndcg_10 = 0
        for epoch in range(epoch_num):
            print('***** Epoch {} *****\n'.format(epoch + 1))
            start = time.time()
            epoch_loss = 0
            for i, batch in enumerate(train_loader):
                texts = batch['text']
                labels = batch['label']
                tokenized_text = self.tokenizer(texts, max_length=self.MAX_TEXT_LENGTH, truncation=True,
                                                padding='max_length', return_tensors='pt')
                # 文本长度不足MAX_TEXT_LENGTH的部分padding填充

                outputs = model(**tokenized_text.to(device))  # **是指参数以字典形式传入
                loss = criterion(outputs, labels.view(
                    labels.size(0), 1).float().to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.writer.add_scalar(
                    'loss', loss, global_step=epoch*len(train_loader)+i)
                epoch_loss += loss.item()
                self.writer.add_scalar(
                    'epoch loss', epoch_loss / (i + 1), global_step=epoch*len(train_loader)+i)

                # 计算当前epoch的平均训练损失epoch_loss / (i + 1)
                if (i + 1) % (len(train_loader) // 10) == 0:
                    print('Step {:04d}/{:04d}  Epoch Loss: {:.4f}  Time: {:.4f}s:'
                          .format((i + 1), len(train_loader), epoch_loss / (i + 1), time.time() - start))

                # 验证集测试
                if (i + 1) % (len(train_loader) // 5) == 0:
                    dev_acc = self.evaluate(device)  # dev=val
                    # dev_ndcg_10 = self.evaluate(device)  # dev=val
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        # current_time = datetime.datetime.now()
                        # print("current_time:    " + str(current_time))
                        # 每次保存的是最优model
                        torch.save(model, '../models/bert_model.pkl')
                    print('dev acc: {:.4f}, best acc: {:.4f}'.format(
                        dev_acc, best_acc))
                    # print('dev_ndcg_10: {:.4f}, best_ndcg_10: {:.4f}'.format(
                    #     dev_ndcg_10, best_ndcg_10))

    # 使用验证集评估模型
    def evaluate(self, device) -> float:
        model = self.model

        model.eval()
        labels_true, labels_pred = [], []   # 存放正确标签和预测标签
        with torch.no_grad():
            for i, batch in (enumerate(self.val_loader)):
                texts = batch['text']
                labels = batch['label']
                tokenized_text = self.tokenizer(texts, max_length=self.MAX_TEXT_LENGTH, truncation=True,
                                                padding='max_length', return_tensors='pt')
                outputs = model(**tokenized_text.to(device))
                outputs_np = outputs[:, 0].detach().cpu().numpy()
                # 概率>0.5则为1，否则为0
                labels_pred.extend(np.where(outputs_np > 0.5, 1, 0).tolist())
                labels_true.extend(labels.squeeze().cpu().numpy().tolist())

        # return ndcg_score(labels_true, labels_pred, 10)  # 返回ndcg@10
        return accuracy_score(labels_true, labels_pred)  # 返回acc
