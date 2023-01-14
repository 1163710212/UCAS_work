import torch.nn as nn
from transformers import BertModel
import torch


class BertRegressionModel(nn.Module):
    def __init__(self, seg_len, device):
        super(BertRegressionModel, self).__init__()
        self.seg_len = seg_len
        # 加载预训练模型
        pretrained_weights = "bert-base-chinese"
        self.bert = BertModel.from_pretrained(pretrained_weights)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义线性函数
        self.dense = nn.Sequential(nn.Linear(768 * self.seg_len, 768),
                                   nn.ReLU(),
                                   nn.LayerNorm(768),
                                   nn.Linear(768, 768),
                                   nn.ReLU(),
                                   nn.LayerNorm(768),
                                   nn.Linear(768, 1)) 
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        # 得到bert_output
        out_l = []
        batch_size = input_ids.shape[0]
        input_ids = input_ids.transpose(1, 0)
        token_type_ids = token_type_ids.transpose(1, 0)
        attention_mask = attention_mask.transpose(1, 0)
        for i in range(self.seg_len):
            bert_output = self.bert(input_ids=input_ids[i], token_type_ids=token_type_ids[i],
                                    attention_mask=attention_mask[i], output_hidden_states=True)
            # 获得预训练模型的输出
            bert_cls_hidden_state = bert_output[0][:, 0, :]
            out_l.append(bert_cls_hidden_state)
        # embedding_out = torch.cat(out_l, axis=1).view(batch_size, self.seg_len, -1)
        embedding_out = torch.cat(out_l, axis=1).view(batch_size, -1)
        # 将768维的向量输入到线性层映射为一维向量
        # time = self.predict(embedding_out)
        time = self.dense(embedding_out)
        return time