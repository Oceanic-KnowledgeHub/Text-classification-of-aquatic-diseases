import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    '''
    最简单的自注意力模块
    '''
    def __init__(self, d_model ,d_head):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.w_q = nn.Linear(d_model, d_head)
        self.w_k = nn.Linear(d_model, d_head)
        self.w_v = nn.Linear(d_model, d_head)
        self.w_o = nn.Linear(d_head, d_model)
    def forward(self, x):
        '''
        :param x: 输入文本的embedding [batchsize, seq_len, d_model]
        :return: 自注意力结果
        '''
        #首先得到q,k,v
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # 计算attention q[bsz, seq_len, d_model]
        # K的维度交换一下
        attn_score = torch.matmul(q, k.permute(0, 2, 1))
        # 点积结果 -- softmax
        attn_score = torch.softmax(attn_score, dim=-1)
        # 注意力权重与value相乘
        output = torch.matmul(attn_score, v)
        # 输出矩阵 [bsz, seq_len, d_model]
        output = self.w_o(output)

        return output

# if __name__ == '__main__':
#     x = torch.randn(4, 12, 768)
#     print(x)
#     self_attn = SelfAttention(d_model=768, d_head=64)
#     out = self_attn(x)
#     # print(out)
#     print(out.shape)