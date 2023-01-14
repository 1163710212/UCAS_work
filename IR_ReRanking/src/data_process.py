import pandas as pd

# 读取训练集
# passages
passages = pd.read_csv(
    "data/collection.train.sampled.tsv", sep="\t", header=None)
passages.columns = ['pid', 'passage']
passages.set_index('pid', inplace=True)

# queries
queries = pd.read_csv("data/queries.train.sampled.tsv", sep="\t", header=None)
queries.columns = ['qid', 'query']
queries.set_index('qid', inplace=True)

# triples （查询与正负样本）
triples = pd.read_csv(
    "data/qidpidtriples.train.sampled.tsv", sep="\t", header=None)
triples.columns = ['qid', 'pos_pid', 'neg_pid']
# 生成训练集
# 将原始数据合并，使其成为模型的输入

# 正负样本
train_set_pos = pd.DataFrame(
    data=None, columns=['query', 'passage', 'is_relevant'])
train_set_pos['query'] = triples['qid']
train_set_pos['passage'] = triples['pos_pid']
train_set_pos['is_relevant'] = 1

train_set_neg = pd.DataFrame(
    data=None, columns=['query', 'passage', 'is_relevant'])
train_set_neg['query'] = triples['qid']
train_set_neg['passage'] = triples['neg_pid']
train_set_neg['is_relevant'] = 0

# 合并正负样本
train_set = pd.concat([train_set_pos, train_set_neg],
                      axis=0, ignore_index=True)
train_set = train_set.sample(frac=1).reset_index(drop=True)  # 打乱顺序

# 将id替换为具体文本
train_set['query'] = train_set['query'].apply(
    lambda x: queries.loc[x, 'query'])
train_set['passage'] = train_set['passage'].apply(
    lambda x: passages.loc[x, 'passage'])

# 保存训练集
train_set.to_csv("data/train_set.tsv", sep="\t", index=False)
print("训练集生成完毕，train_set已保存至data目录下！")
