import pandas as pd
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
from model import BertRegressionModel
from prepare_data import load_data
import argparse


def predict(model, test_loader, device):
    model.to(device)
    model.eval()
    predicts = []
    with torch.no_grad():
        for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(test_loader):
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)
            out_put = model(input_ids, token_type_ids, attention_mask).view(-1)
            predicts.extend(out_put.data.tolist())
            if (step + 1) % 1000 == 0:
                print(f"{step} complete")
        # 返回预测结果
        return predicts

# 设定batch_size
batch_size = 8
test_data_path = "../raw_data/testA.csv"
test_data = load_data(test_data_path, False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# 引进训练好的模型进行测试
parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--model_path', type=str, default='')
args = parser.parse_args()
fold = args.fold
cuda = args.cuda
model_path = args.model_path
path = f'../user_data/model/{model_path}'
model = torch.load(path)
device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')

# 预测的减刑时间
predicts = predict(model, test_loader, device)

# 将结果写入文件
submission = pd.concat([pd.read_csv(test_data_path).iloc[:, 0], pd.DataFrame(predicts, columns=['label'])], axis=1)
submission.to_csv(f'../user_data/submission/submission_fold{fold}.csv', index=False, header=True)