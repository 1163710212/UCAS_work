import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# 选取众数
def majorityLabel(labels):
    return Counter(list(labels)).most_common()[0]

df_list = []
for i in range(1, 8):
    df = pd.read_csv(f'../user_data/submission/submission_fold{i}.csv')
    df.iloc[:, 1] = df.iloc[:, 1]
    df_list.append(df)

# 众数出现次数大于1，选众数，否则取均值
submission = df_list[2].copy()
c = 0
for i in range(submission.shape[0]):
    time_l = [int(np.round(df.iloc[i, 1])) for df in df_list]
    time_f_l = np.array([df.iloc[i, 1] for df in df_list])
   # print(majorityLabel(time_l), time_l)
    maj_res = majorityLabel(time_l)
    if maj_res[1] >= 4:
        c += 1
        submission.iloc[i, 1] = maj_res[0]
    else:
        submission.iloc[i, 1] = np.average(time_f_l)
    #submission.iloc[i, 1] = np.average(time_f_l)

print(c)
submission.iloc[:, 1] = submission.iloc[:, 1].round().astype(int)
submission.to_csv('../prediction_result/result.csv', header=True, index=False)
