import torch 
"""
scaler = torch.tensor(5)
print(scaler.ndim)
vector = torch.tensor([1,2])
print(vector.ndim)
"""
#x = torch.tensor([1,2,3,4,5,6])
#x = torch.zeros(2,3)
#x = torch.ones(2,1)
#x = torch.rand(2,2)
#x = torch.randn(2,2)
#print(x)
#print(torch.Size([3,5]))
#x = torch.tensor([1,2,3])
#print(x.dtype)

"""
a = torch.tensor([1,2,3])
b = torch.tensor([5,6,7])
print(a + b)
print(a - b)
print(a / b)
print(a * b)

a = torch.tensor([1,2,3])
print(a.min())
print(a.max())
print(a.sum())
"""
"""
import numpy as np
arr = torch.tensor([1,2,3])
np_arr = arr.numpy()
print(np_arr)

arr = np.array([1,2,3])
print(arr.dtype)
x = torch.from_numpy(arr)
print(x.dtype)
"""

x = torch.tensor([5,10,15,20])
y = torch.zeros(3,4)
z = torch.rand(5,5)
print(z.shape)
print(z.dtype)
print(z.ndim)

a = torch.tensor([10,20,30])
b = torch.tensor([1,2,3])
print(a+b)
print(a-b)
print(a*b)
print(a/b)

rands = torch.rand(1000)
print(rands.mean())
print(rands.std())
print(rands.max())
print(rands.min())