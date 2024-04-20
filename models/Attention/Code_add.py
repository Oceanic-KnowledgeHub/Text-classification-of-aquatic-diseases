import torch
import torch.nn as nn
import torch.nn.functional as F

class Code_add(nn.Module):
    def __init__(self, d_model ,d_head):
        super(Code_add, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.q = nn.Linear(d_model * 2, d_head)
        self.t = nn.Linear(d_model, d_head)
        self.w_o = nn.Linear(32, d_model)

    def forward(self, x1, x2):
        x1 = self.q(x1)
        x2 = self.t(x2)
        #x3 = torch.matmul(x1, x2.permute(0, 2, 1))
        x4 = torch.matmul(x2, x1.permute(0, 2, 1))
        #x5 = torch.softmax(x3, dim=-1)
        x6 = torch.softmax(x4, dim=-1)
        #x = x5 + x6
        # x = F.relu(x6)
        output = self.w_o(x6)
        return output

