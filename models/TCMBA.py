# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
from torch.nn import init
import torch.nn.functional as F
from models.Attention.SelfAttention import SelfAttention
from models.Attention.Code_add import Code_add
from transformers import BertTokenizer, ErnieForMaskedLM

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'TCMBA'
        self.train_path = dataset + '/data/train1.txt'
        self.dev_path = dataset + '/data/dev1.txt'
        self.test_path = dataset + '/data/test1.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class1.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 30
        self.batch_size = 32
        self.pad_size = 32
        self.learning_rate = 3e-5
        self.bert_path = './ERNIE3.0_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2
        # self.droplast = true

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        #词嵌入层
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        #lstm层
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        #注意力
        self.attn = SelfAttention(d_model=config.hidden_size * 2, d_head=config.batch_size)
        # 融合特生成权重，加入CNN前期
        self.encoder_add = Code_add(d_model=config.hidden_size, d_head=config.batch_size)
        #卷积层
        self.convs = nn.ModuleList(
             [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        #输出层
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 3, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        #词嵌入层
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # lstm层
        out, _ = self.lstm(encoder_out)
        #注意力层
        out = self.attn(out)
        #融合特生成权重，加入CNN前期
        encoder_out_last = self.encoder_add(out, encoder_out)
        #卷积层
        out_cnn = encoder_out_last.unsqueeze(1)
        out_cnn = torch.cat([self.conv_and_pool(out_cnn, conv) for conv in self.convs], 1)
        #输出层
        out = torch.cat((out[:, -1, :],out_cnn),1)
        out = self.dropout(out)
        out = self.fc(out)

        return out



