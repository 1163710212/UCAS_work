import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import torch
import time
import translators as ts


# 进行数据增强
def enhance_data():
    train_data = pd.read_csv(f'../user_data/data/train.csv')
    trans_train_data = train_data.copy #pd.read_csv(f'./data/enhance.csv')
    while (True):
        try:
            for i in range(train_data.shape[0]):
                if train_data.iloc[i, 0] != trans_train_data.iloc[i, 0]:
                    continue
                org_text = train_data.iloc[i, 0]
                if len(org_text) > 1000:
                    org_text = org_text[:500] + org_text[-500:]
                if (i + 1) % 10 == 0:
                    print(f'{i} step complete')
                    trans_train_data.to_csv(f'./data/enhance.csv', index=False, header=True)
                en_text = ts.baidu(org_text, if_use_cn_host=True, from_language='zh', to_language='en')
                trans_text = ts.baidu(en_text, if_use_cn_host=True, from_language='en', to_language='zh')
                trans_train_data.iloc[i, 0] = trans_text
        except Exception:
            time.sleep(5)
            print("Intercept")
            
# 生成K fold数据集
def generate_fold_data():
    train_data = pd.read_csv(f'../user_data/data/train.csv')
    enhance_data = pd.read_csv(f'../user_data/data/enhance.csv')
    
    base_path = '../user_data/data/fold/'
    pd.concat([train_data.iloc[:45000], enhance_data.iloc[:45000]], axis=0).to_csv(f'{base_path}train_f1.csv', header=True,
                                                                                   index=False)
    train_data.iloc[45000:].to_csv(f'{base_path}dev_f1.csv', header=True, index=False)

    pd.concat([train_data.iloc[5000:], enhance_data.iloc[5000:]], axis=0).to_csv(f'{base_path}train_f2.csv', header=True,
                                                                                 index=False)
    train_data.iloc[:5000].to_csv(f'{base_path}dev_f2.csv', header=True, index=False)

    for i in range(3, 11):
        head = (i - 2) * 5000
        tail = (i - 1) * 5000
        pd.concat([train_data.iloc[:head],
                   train_data.iloc[tail:],
                   enhance_data.iloc[:head],
                   enhance_data.iloc[tail:]], axis=0).to_csv(f'{base_path}train_f{i}.csv', header=True, index=False)
        train_data.iloc[head:tail].to_csv(f'{base_path}dev_f{i}.csv', header=True, index=False)
    
#### 使用BertTokenizer 编码成Bert需要的输入格式
def encoder(max_len, vocab_path, text_list, p_num=2):
    # 将text_list embedding成bert模型可用的输入形式
    # 加载分词模型
    tokenizer = AutoTokenizer.from_pretrained(vocab_path)
    tokenizer = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'  # 返回的类型为pytorch tensor
    )
    seg_len = tokenizer['input_ids'].shape[1]
    input_ids = tokenizer['input_ids'].view(-1, p_num, seg_len)
    token_type_ids = tokenizer['token_type_ids'].view(-1, p_num, seg_len)
    attention_mask = tokenizer['attention_mask'].view(-1, p_num, seg_len)
    return input_ids, token_type_ids, attention_mask


#### 将数据加载为Tensor格式
def load_data(path, is_train=True):
    if is_train:
        data = pd.read_csv(path, encoding='utf-8')
    else:
        data = pd.read_csv(path, encoding='utf-8').iloc[:, 1:]
    print(data.shape)
    text_list_org = data.iloc[:, 0].tolist()
    lengths = []
    text_list = []
    for text in text_list_org:
        lengths.append(min(3, int(len(text) / 500)))
        if len(text) >= 1000:
            text = text[:500] + text[-500:]
        else:
            text += "[PAD]" * (1000 - len(text))
        text_list.extend([text[:500], text[500: 1000]])
    if is_train:
        labels = data.iloc[:, 1].tolist()
    else:
        labels = [0] * data.iloc[:].shape[0]
    # 调用encoder函数，获得预训练模型的三种输入形式
    input_ids, token_type_ids, attention_mask = encoder(max_len=512, vocab_path="bert-base-chinese",
                                                        text_list=text_list)
    labels = torch.tensor(labels, dtype=torch.float)
    # 将encoder的返回值以及label封装为Tensor的形式
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    return data

if __name__ == "__main__":
    # enhance_data()
    generate_fold_data()