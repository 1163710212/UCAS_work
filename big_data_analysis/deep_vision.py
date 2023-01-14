import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython import display

display.set_matplotlib_formats('svg')

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

# 获取数据
back_develop = pd.read_csv(f'data/dataTrain.csv')
print(type(back_develop))
# 查看数据大小
print(back_develop.shape)
back_develop.head(10)

# x_train =  back_develop.iloc[:,1:47]
x_train = back_develop[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8',
                        'f18', 'f19', 'f21', 'f23', 'f24', 'f25', 'f26',
                        'f30', 'f37', 'f43', 'f44', 'f45', 'f46']]
y_train = back_develop.iloc[:, -1]
print(x_train.shape, y_train.shape)

# 进行处理（特征工程）特征-》类别-》one_hot编码
dict = DictVectorizer(sparse=False)
# 进一步对字典进行特征抽取，to_dict可以把df变成字典，records代表列名变为键
x_train = dict.fit_transform(x_train.to_dict(orient="records"))
print(x_train.shape)

# tf.keras.models.Sequential()
# 构造模型
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[1, 23]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation="selu"))
model.add(keras.layers.AlphaDropout(rate=0.2))
model.add(keras.layers.Dense(1, activation="sigmoid"))

split_point = int(x_train.shape[0] * 0.8)
x_valid, y_valid = x_train[split_point:], y_train[split_point:]
x_train, y_train = x_train[:split_point], y_train[:split_point]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(name='auc')])
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-5),
]
# 模型训练
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=20)


def plot_auc(history):
    # Use a log scale to show the wide range of values.
    plt.plot(history.epoch, history.history['auc'], label='Train Auc')
    plt.plot(history.epoch, history.history['val_auc'], label='Val Auc', linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    print(history.epoch)
    plt.yticks(np.linspace(0.5, 1, 11))
    plt.xticks(history.epoch + 1)
    plt.legend()
    plt.savefig('deep_auc.svg', type='svg')


plot_auc(history)
