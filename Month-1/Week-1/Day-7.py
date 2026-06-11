import torch
import torch.nn as nn

#Project 1: Linear Regression
"""
X = torch.tensor([
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
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    print(epoch, loss.item())

with torch.no_grad():
    print(model(torch.tensor([[4.0]])))
"""

#Project 2: Binary Classification
"""
X = torch.tensor([
    [30.0],
    [40.0],
    [50.0],
    [60.0],
    [70.0],
    [80.0]
])

y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])

model = nn.Sequential(
    nn.Linear(1,1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    print(epoch, loss.item())
    
with torch.no_grad():
    print(model(torch.tensor([[40.0]])))
"""

#Multi-Class Classification
"""
X = torch.tensor([
    [1.0,1.0],
    [1.0,2.0],
    [2.0,1.0],

    [5.0,5.0],
    [5.0,6.0],
    [6.0,5.0],

    [9.0,9.0],
    [9.0,10.0],
    [10.0,9.0]
])

y = torch.tensor([
    0,
    0,
    0,

    1,
    1,
    1,

    2,
    2,
    2
])

model = nn.Sequential(
    nn.Linear(2,16),
    nn.ReLU(),
    nn.Linear(16,3)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(1000):
    output = model(X)
    loss = criterion(
        output, y
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch, loss.item())
    
with torch.no_grad():
    test = torch.tensor([[3.4,3.4]])
    logits = model(test)
    prediction = torch.argmax(
        logits, dim=1
    )    
    print(prediction)
"""