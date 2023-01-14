import os

import torch
import pandas as pd
from transformers import BertTokenizer
from models import TextClassifier
from trainer import Trainer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, '..')
# os.path.join(ROOT_DIR, 'models/pretrained/bert-base-uncased')
BERT_DIR = 'bert-base-uncased'
# https://huggingface.co/bert-base-uncased

# Init tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
# Init model
model = TextClassifier(BERT_DIR).to(DEVICE)
# Init dataset
dataset = pd.read_csv(os.path.join(ROOT_DIR, 'data/train_set.tsv'), sep='\t')

# 获取query和passage文本连接最大长度，赋值给MAX_TEXT_LENGTH = 226(224+2,加上界定符CLS、SEP)
max_len = 0
for i in range(dataset.shape[0]):
    # print(max_len)
    max_len = max(
        len((dataset.loc[i, 'query'] + dataset.loc[i, 'passage']).split(' ')), max_len)
print(f"Max text length：{max_len}")

# Init trainer
trainer = Trainer(model, tokenizer, dataset)
# Train with device and epoch_num
trainer.train(DEVICE, 5)

print("训练完毕，bert_model已保存至../models目录下！")

