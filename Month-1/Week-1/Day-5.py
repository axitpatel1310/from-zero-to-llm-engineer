#Optimizers, SGD and Adam

"""
What is Optimizer
- An optimizer updates model parameters using gradients
"""

import torch
"""
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(10.0)
target = torch.tensor(20.0)
prediction = w * x
loss = (prediction - target) ** 2
loss.backward()
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()
print(w)
"""

#Using SGD Optimizer
"""
w = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.SGD(
    [w],
    lr=0.01
)
x = torch.tensor(10.0)
target = torch.tensor(20.0)
prediction = w * x
loss = (prediction - target) ** 2
loss.backward()
optimizer.step()
print(w)
"""

#Mini Build
"""
w = torch.tensor(
    1.0,
    requires_grad=True
)
optimizer = torch.optim.SGD(
    [w],
    lr=0.01
)
x = torch.tensor(10.0)
target = torch.tensor(20.0)

for step in range(10):
    optimizer.zero_grad()
    prediction = w*x
    loss = (prediction-target)**2
    loss.backward()
    optimizer.step()
    print(step, w.item(),loss.item())    
"""

#Mini Build 2
"""
w = torch.tensor(
    1.0,
    requires_grad=True
)
optimizer = torch.optim.Adam(
    [w],
    lr=0.01
)
x = torch.tensor(10.0)
target = torch.tensor(20.0)

for step in range(10):
    optimizer.zero_grad()
    prediction = w*x
    loss = (prediction-target)**2
    loss.backward()
    optimizer.step()
    print(step, w.item(),loss.item())    
"""

#Exercise 1
"""
w = torch.tensor(
    3.0, requires_grad=True
)
x = torch.tensor(1.0)
loss = (w - 10)**2
target = torch.tensor(10.0)

optimizer = torch.optim.Adam(
    [w],
    lr=0.01
)
for i in range(30):
    optimizer.zero_grad()
    prediction = w * x
    loss = (prediction - target) ** 2
    loss.backward()
    optimizer.step()
    print(i, w, loss.item())
"""

#Exercise 2
"""
model = torch.nn.Linear(
    3,1
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
for param in model.parameters():
    print(param)
"""
 