import os

import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import transformers

# print(torch.__version__)
# print(pd.__version__)
# print(np.__version__)
# print(transformers.__version__)
# exit

# Init tokenizer


class TextSet(Dataset):
    def __init__(self, dataset_df):
        self.dataset = dataset_df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        query = self.dataset.loc[index, 'query']
        passage = self.dataset.loc[index, 'passage']
        return {'text': query + '[SEP]' + passage}


def predict(dataset) -> list:
    tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
    model = torch.load(CURRENT_MODEL_FILE, map_location=DEVICE)
    model.eval()
    data_loader = DataLoader(TextSet(dataset), batch_size=8, shuffle=False)

    scores = []
    for i, data in enumerate(data_loader):
        text = data['text']
        tokenized_text = tokenizer(text, max_length=MAX_TEXT_LENGTH, truncation=True, padding=True, return_tensors='pt').to(
            DEVICE)
        outputs = model(**tokenized_text)
        scores.extend(outputs.cpu().detach().numpy().tolist())
        if (i + 1) % 500 == 0:
            print(f'{i + 1} steps finished')
    return scores


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_TEXT_LENGTH = 226

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, '..')
# os.path.join(ROOT_DIR, 'models/pretrained/bert-base-uncased')
BERT_DIR = 'bert-base-uncased'
CURRENT_MODEL_FILE = os.path.join(ROOT_DIR, 'models/bert_model.pkl')

# 读取验证集（使用2019年的测试集作为验证集，43个验证查询），本次任务没用到
# val_set
val_set = pd.read_csv(os.path.join(
    ROOT_DIR, 'data/msmarco-passagetest2019-43-top1000.tsv'), sep='\t', header=None)
val_set.columns = ['qid', 'pid', 'query', 'passage']
# print(val_set['qid'].value_counts().count())

# 读取测试集（2020年的测试集，54个测试查询）
# test_set
test_set = pd.read_csv(os.path.join(
    ROOT_DIR, 'data/msmarco-passagetest2020-54-top1000.tsv'), sep='\t', header=None)
test_set.columns = ['qid', 'pid', 'query', 'passage']
# print(test_set['qid'].value_counts().count())

# if __
# test_set = test_set.iloc[:2000]
test_set['q0'] = 'QO'
test_set['sid'] = 'I_LIKE_IR'
test_set['score'] = np.array(predict(test_set)) * 10
test_set = test_set.sort_values(by='score', ascending=False, axis=0)
test_set['rank'] = np.arange(1, test_set.shape[0] + 1)

# dir_name = '../trec_eval'
# if not os.path.isdir(dir_name):
#     os.makedirs(dir_name)
# test_set[['qid', 'q0', 'pid', 'rank', 'score', 'sid']].to_csv(
#     '../trec_eval/res_BERT', header=False, index=False, sep='\t')
test_set[['qid', 'q0', 'pid', 'rank', 'score', 'sid']].to_csv(
    'res_BERT', header=False, index=False, sep='\t')

print("重排完毕，res_BERT结果文件已保存至工程根目录下！")
