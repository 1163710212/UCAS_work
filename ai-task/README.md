# 竞赛：罪犯减刑预测
地址 https://www.datafountain.cn/competitions/611

## 1.环境与依赖的python库

- Linux version: 3.10.0-1160.41.1.el7.x86_64
- CUDA Version: 11.7 
- 需要安装的python库：numpy(1.19.2),pandas(1.1.5),torch(1.4.0),tqdm(4.64.1),transformers(3.3.0)、matplotlib、ipython(6.5.0)

## 2.数据分析
- AnalyseData.ipynb，分析文本长度分布、时间分布

## 3.训练数据准备
- code/prepare_data.py下的enhance_data()为用翻译，再回译方式生成增强数据的方法
- code/prepare_data.py下的generate_fold_data()生成k折训练集、验证集的方法
- user_data/data/train.csv为原训练文件去掉id列，随机打乱后的结果
- user_data/data/enhance.csv为生成的增强数据
- user_data/data/fold下为k折训练集、验证集

## 4.模型训练
- 将终端路径切换到code文件夹下，运行命令：python tran.py --cuda [cuda_num] --fold [fold_num]
- 示例：python tran.py --cuda 0 --fold 1 为运用第0张gpu训练第1折数据
- 训练过程的日志文件保存在user_data/log文件夹下

## 5.生成预测数据
- 将终端路径切换到code文件夹下，运行命令：python predict.py --cuda [cuda_num] --fold [fold_num] --mode_path [mode_name]
- 示例：python tran.py --cuda 0 --fold 1 --mode_path 'bert_fold1_0.pkl' 为运用第0张gpu,使用第1折数据训练的模型bert_fold1_0.pkl
  预测
  

##  6.生成最终提交文件

- 将终端路径切换到code文件夹下，运行命令：python generate_sub_data.py
- 由于最后时间不够，只训练了7折数据。剩下3折数据也训练后，在将其模型预测结果结合进来，应该会有更好的效果。