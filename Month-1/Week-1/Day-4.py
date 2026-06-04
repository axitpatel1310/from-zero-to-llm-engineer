#Computational Graphs, Gradients and Backward

"""
Forward Pass
      ↓
Calculate Loss
      ↓
Compute Gradients
      ↓
Update Weights
      ↓
Repeat
"""
import torch

#Creating a Tensor that track gradients
"""
x = torch.tensor(2.0)
print(x.requires_grad)
#Enable Gradients
x = torch.tensor(2.0, requires_grad=True)
print(x.requires_grad)
"""

#Computational Graph
"""
x = torch.tensor(
    2.0,
    requires_grad=True
)
y = x ** 2
y.backward()
print(x.grad)
# In torch when we say require grad is tracks the grads but not give it as output until backward is used.
"""

#Chain Rule
"""
x = torch.tensor(
    2.0, 
    requires_grad=True
)
a = x * 2
y = a ** 2
y.backward()
print(x.grad)
"""

#Gradient Accumulate
"""
x = torch.tensor(
    2.0,
    requires_grad=True
)
y = x ** 2
y.backward()
print(x.grad)
y = x ** 2
y.backward()
print(x.grad)

#Resetting gradients
x.grad.zero_()
print(x.grad)
"""

#Multiple Variables
"""
x = torch.tensor(
    2.0,
    requires_grad=True
)
w = torch.tensor(
    3.0,
    requires_grad=True
)
y = x * w
y.backward()
print(x.grad)
print(w.grad)
"""

#Mini Build 1
"""
x = torch.tensor(
    4.0,
    requires_grad=True
)
y = x**2 + 3*x
y.backward()
print(x.grad)
"""

#Mini Build 2
"""
w = torch.tensor(
    2.0, requires_grad=True
)
x = torch.tensor(5.0)
prediction = w*x
prediction.backward()
print(w.grad)
"""

#Exercise 1
"""
x = torch.tensor(
    5.0,
    requires_grad=True
)
y = x**2
y.backward()
print(x.grad)
"""

#Exercise 2
"""
x = torch.tensor(
    2.0,
    requires_grad=True
)
y = x**3
y.backward()
print(x.grad)
"""

#Exercise 3
"""
x = torch.tensor(
    4.0, 
    requires_grad=True
)
y = 5*x + 7
y.backward()
print(x.grad)
"""

#Exercise 4
"""
x = torch.tensor(
    3.0,
    requires_grad=True
)
w = torch.tensor(
    2.0,
    requires_grad=True
)
y = x*w
y.backward()
print(x.grad)
print(w.grad)
"""

#Exercise 5
"""
w = torch.tensor(
    1.0, 
    requires_grad=True
)
x = torch.tensor(10.0)
prediction = w * x
loss = (prediction - 20)**2
loss.backward()
print(w.grad)
"""