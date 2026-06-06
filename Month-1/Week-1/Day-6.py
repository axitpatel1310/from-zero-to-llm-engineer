#Complete Trainning Loop
"""
1. Forward Pass
2. Compute Loss
3. Zero Gradients
4. Backward Pass
5. Update Parameters
"""
import torch
import torch.nn as nn
"""
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])
y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])

model = nn.Linear(
    in_features=1,
    out_features=1
)
#print(model.weight)
#print(model.bias)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
for epoch in range(200):
    prediction = model(x)
    loss = criterion(
        prediction, y
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(epoch, loss.item())
        
with torch.no_grad():
    prediction = model(
        torch.tensor([[5.0]])
    )
print(prediction)
"""

#Mini Build 1: Linear Regression
"""
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0]
])

model = nn.Linear(1,1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    prediction = model(x)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(epoch, loss.item())
        
with torch.no_grad():
    prediction = model(torch.tensor([[4.0]]))
    print(prediction)    
"""

#Mini Build 2: Noisy data
"""
x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [2.1],
    [3.9],
    [6.2],
    [7.8],
    [10.1]
])

model = nn.Linear(1,1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    prediction = model(x)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())

with torch.no_grad():
    prediction = model(torch.tensor([[2.0]]))
    print(prediction)
"""

#Final Mini Project
"""
x = torch.randn(100,3)
y = (
    2*x[:,0]
    -3*x[:,1]
    +4*x[:,2]
).unsqueeze(1)

model = nn.Linear(3,1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    prediction = model(x)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 100:
        print(epoch, loss.item())
        
with torch.no_grad():
    print(model(torch.tensor([[30.0,29.0,19.0]])))
"""    