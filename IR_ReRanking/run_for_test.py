import os

print("*****************************************获取train_set数据集*************************************")
os.system("python ./src/data_process.py")
print("*****************************************begin training*****************************************")
os.system("python ./src/main.py")
print("*****************************************begin reranking****************************************")
os.system("python ./src/rank.py")
print("*****************************************NDCG结果如下********************************************")
os.system(
    "cd ./trec_eval ; trec_eval -m ndcg_cut ../data/2020qrels-pass.txt ../res_BERT")
