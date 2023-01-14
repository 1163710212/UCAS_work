import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
# 数据读取
data_of_trian = pd.read_csv(f'data/dataTrain.csv')[:50000]
from sklearn.utils import shuffle
data_of_trian = shuffle(data_of_trian)

data_of_test = pd.read_csv(f'data/dataA.csv')
submission = pd.read_csv(f'data/submit_example_A.csv')
data_nolabel = pd.read_csv(f'data/dataNoLabel.csv')

# 查看一下数据的shape
print(f'data_of_trian.shape = {data_of_trian.shape}')
print(f'data_of_test.shape  = {data_of_test.shape}')

# ID类特征数值化
cat_columns = ['f3']
data = pd.concat([data_of_trian, data_of_test])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    data_of_trian[col] = lb.transform(data_of_trian[col])
    data_of_test[col] = lb.transform(data_of_test[col])
# 最后构造出训练集和测试集
num_columns = [col for col in data_of_trian.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = data_of_trian[feature_columns]
label = data_of_trian[target]
test = data_of_test[feature_columns]


# 定义交叉验证模型框架,采用五折交叉验证
# 传入模型，输出各模型AUC计算值
def model_of_trian(model, name_of_model, kfold=5):
    predict_oof = np.zeros((train.shape[0]))
    predict_test = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model_name = {name_of_model}")
    valid_auc_l, train_auc_l = [], []
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train, y_train)

        y_predict = model.predict_proba(x_test)[:, 1]
        predict_oof[test_index] = y_predict.ravel()
        # 计算AUC
        auc = roc_auc_score(y_test, y_predict)
        train_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
        # 打印一下交叉验证次数和AUC指标
        print("----KFold = %d, val_auc = %.4f" % (k, auc))
        valid_auc_l.append(auc)
        train_auc_l.append(train_auc)
        test_fold_preds = model.predict_proba(test)[:, 1]
        predict_test += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (name_of_model, roc_auc_score(label, predict_oof)))
    return predict_test / kfold, valid_auc_l, train_auc_l


# 数据清洗
# 将数据切割成50份，对每一份数据单独作为验证集，
# 如果验证集的AUC几乎接近于0.5，则验证集中的数据大多数为干扰数据
# 这部分运行时间太长，建议不打开运行
# gbc = GradientBoostingClassifier()
# gbc_test_preds = model_of_trian(gbc, "GradientBoostingClassifier", 50)

# 很明显可以看出，50~59的数据为干扰数据
# 所以剔除干扰数据
train = train[:50000]
label = label[:50000]

# 模型融合
# GradientBoostingClassifier
Gbc = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=6
)
# HistGradientBoostingClassifier
# HGbc = HistGradientBoostingClassifier(
#     max_iter=100,
#     max_depth=6
# )
# XGBClassifier
XGbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1
)
# LGBMClassifier
LGbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6,
    max_depth=10,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.05,
    n_estimators=100,
    metrics='auc'
)
# CatBoostClassifier
Cbc = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=1,
    loss_function='Logloss',
    verbose=0
)

estimators = [
    ('gbc', Gbc),
    # ('hgbc', HGbc),
    ('xgbc', XGbc),
    ('gbm', LGbm),
    ('cbc', Cbc)
]
# 通过StackingClassifier将6个模型进行堆叠（Stack）
# Stack模型用LogisticRegression
model_fusion = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
model_fusion = LGbm

# 先将训练数据划分成训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)
# 然后用组合模型进行训练和验证
model_fusion.fit(X_train, y_train)
y_pred = model_fusion.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.6f' % auc)

# 特征工程
# 感觉特征偏少，自己进行特征构造
# 这个f47是我自己简单构造的，把特征f1*10加上特征f2
data_of_trian['f47'] = data_of_trian['f1'] * 10 + data_of_trian['f2']
data_of_test['f47'] = data_of_test['f1'] * 10 + data_of_test['f2']
# Feature 位置特征
f_location = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [data_of_trian, data_of_test]:
    for i in range(len(f_location)):
        for j in range(i + 1, len(f_location)):
            df[f'{f_location[i]}+{f_location[j]}'] = df[f_location[i]] + df[f_location[j]]
            df[f'{f_location[i]}-{f_location[j]}'] = df[f_location[i]] - df[f_location[j]]
            df[f'{f_location[i]}*{f_location[j]}'] = df[f_location[i]] * df[f_location[j]]
            df[f'{f_location[i]}/{f_location[j]}'] = df[f_location[i]] / (df[f_location[j]] + 1)

# Feature 通话特征
call_f = ['f43', 'f44', 'f45', 'f46']
for df in [data_of_trian, data_of_test]:
    for i in range(len(call_f)):
        for j in range(i + 1, len(call_f)):
            df[f'{call_f[i]}+{call_f[j]}'] = df[call_f[i]] + df[call_f[j]]
            df[f'{call_f[i]}-{call_f[j]}'] = df[call_f[i]] - df[call_f[j]]
            df[f'{call_f[i]}*{call_f[j]}'] = df[call_f[i]] * df[call_f[j]]
            df[f'{call_f[i]}/{call_f[j]}'] = df[call_f[i]] / (df[call_f[j]] + 1)
# 循环遍历特征，对验证集中的特征进行mask
# 将效果好的特征添加到特征tf这个空的数组
tf = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, model_fusion.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        tf.append(col)
    print('%6s || %.6f || %.6f' % (col, auc1, auc1 - auc))
# 这里选取所有差值为负的特征，对比特征筛选后的特征提升
model_fusion.fit(X_train[tf], y_train)
y_pred = model_fusion.predict_proba(X_test[tf])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.6f' % auc)

# 筛选出重要特征列进行训练和预测
train = train[tf]
test = test[tf]

# 预测,采用10折交叉验证。
test_predicts, valid_auc_l, train_auc_l = model_of_trian(model_fusion, "StackingClassifier", 10)


def plot_auc(train_auc, valid_auc, fold):
    # Use a log scale to show the wide range of values.
    fold = np.arange(fold) + 1
    plt.plot(fold, train_auc, label='Train Auc')
    plt.plot(fold, valid_auc, label='Val Auc', linestyle="--")
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.yticks(np.linspace(0.8, 1, 5))
    plt.xticks(fold)
    plt.legend()
    plt.savefig('rfc_auc.svg', type='svg')


plot_auc(train_auc_l, valid_auc_l, 10)
# submission['label'] = test_predicts
# submission.to_csv('submission.csv', index=False)
