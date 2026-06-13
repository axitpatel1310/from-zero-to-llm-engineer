#Building self attention in Pytorch

import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0], #I 
    [0.0, 2.0, 0.0, 2.0], #Love
    [0.0, 0.0, 1.0, 1.0], #Myself
])

#print(x.shape)

d_model = 4

W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

q = x @ W_q
k = x @ W_k
v = x @ W_v

#print(q.shape)
#print(k.shape)
#print(v.shape)

#Formula to calculate attention
scores = q @ k.T
scores = scores/ (k.shape[-1] ** 0.5)
attention_weights = F.softmax(scores, dim=1)
output = attention_weights @ v

print(output)
"""

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(
            d_model, d_model, bias=False
        )
        self.w_k = nn.Linear(
            d_model, d_model, bias=False
        )
        self.w_v = nn.Linear(
            d_model, d_model, bias=False
        )
    
    def forward(self,x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        scores = Q @ K.transpose(-2,-1)
        scores = scores / (K.size(-1) ** 0.5)
        weights = F.softmax(scores, dim=1)
        output = weights @ V
        return output
    
attention = SelfAttention(128)
x = torch.randn(32, 20, 128)
out = attention(x)
print(out.shape)