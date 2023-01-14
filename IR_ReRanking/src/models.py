from transformers import BertModel, BertConfig
from torch import nn


# BERT + 2层MLP
class TextClassifier(nn.Module):
    def __init__(self, bert_path):
        super(TextClassifier, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        # MLP
        self.predict = nn.Sequential(nn.Linear(self.config.hidden_size, 64),
                                     nn.ReLU(),
                                     nn.LayerNorm(64),
                                     nn.Linear(64, 1),
                                     nn.Sigmoid())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        out = self.predict(out_pool)
        return out
