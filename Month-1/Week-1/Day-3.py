import torch

#Vectors

"""
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
"""

"""
print(a+b)
print(a-b)
print(a*b)
print(a/b)
"""
#print(torch.dot(a,b))
"""
#What happens in the Dot Product
(1×4) + (2×5) + (3×6)
= 4+10+18
= 32
"""
#Both does the exact same thing....
"""
print(torch.matmul(a,b))
print(a @ b)
"""

#Checking Shapes
"""
a = torch.randn(2,3)
b = torch.randn(3,4)
print(a.shape)
print(b.shape)
print(a @ b)
"""

#Matrix Multiplication
"""
features = torch.tensor([
    [1,2],
    [3,4]
], dtype=torch.float32)
weights = torch.tensor([
    [0.5],
    [1.5]
])
output = features @ weights
print(output)
"""

#Broadcasting
"""
x = torch.tensor([
    [1,2,3],
    [4,5,6]
])
print(x+10)
"""

#Broadcasting with Vectors
"""
x = torch.tensor([
    [1,2,3],
    [4,5,6]
])
y = torch.tensor([10,20,30])
print(x + y)
"""

#Vector Similarity
"""
a = torch.tensor([10,20,30])
b = torch.tensor([4,5,6])
similarity = torch.dot(a,b)
print(similarity)
"""

#Simple Neuron
"""
inputs = torch.tensor([
    [2.0,3.0]
    ])
weights = torch.tensor([
    [0.5],
    [1.5]
])
bias = torch.tensor([1.0])
output = torch.matmul(inputs, weights) + bias
print(output)
"""

#Exercise 1
"""
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
print(a+b)
print(a-b)
print(a*b)
print(torch.dot(a,b))
"""

#Exercise 2
"""
a = torch.randn(3,4)
b = torch.randn(4,2)
c = torch.matmul(a,b)
print(c)
print(c.shape)
"""

#Exercise 3
"""
a = torch.ones(5,5)
print(a+10)
"""

#Exercise 4
"""
x = torch.randn(4,3)
y = torch.tensor([1,2,3])
print(x+y)
"""

#Exercise 5
"""
x = torch.tensor([2,3])
y = torch.tensor([3,1])
b = torch.tensor(5)
print(torch.dot(x,y) + b)
"""