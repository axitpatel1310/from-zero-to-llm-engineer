import torch

"""
x = torch.tensor([
    [10,20,30,100],
    [40,50,60,110],
    [70,80,90,120]
])
"""

#print(x)
#print(x[0,0]) # specific element selection
#print(x[0]) # row selection
#print(x[:,0]) #column selection

#Slicing
#print(x[:,:2]) #first args (Rows), second args (Column)
#print(x[:,2:]) 
#print(x[0:2,1:3]) 

#modifiying values
#x[0,1] = 150
#print(x)

#Reshaping
#x = torch.arange(12) #create a scaler
#x = x.reshape(3,4) # convert into matrix
#print(x)

#Flatening
"""
x = torch.tensor([
    [1,2],
    [3,4]
])
flat = x.reshape(-1)
print(flat)
"""

# Using -1 Automatically
"""
x = torch.arange(24)
x = x.reshape(2,-1)
print(x.shape)
"""

#Unsqueeze
"""
x = torch.tensor([1,2,3])
torch.Size([3])
x = x.unsqueeze(0)
print(x.shape)
"""

#Transpose
"""
x = torch.tensor([
    [1,2,3],
    [4,5,6],
])
print(x.T)
"""

#MINI PROJECT
"""
x = torch.arange(1,17)
y = x.reshape(4,4)
print(y[1,:])
print(y[2,:])
print(y[3:,:4])
print(y.reshape(-1))
print(y.reshape(2,8))
print(y.reshape(8,2))
"""

#exercise 1
"""
x = torch.arange(20)
print(x)
x.reshape(4,5)
print(x)
"""

#exercise 2
"""
x = torch.tensor([
    [10,20,30,100],
    [40,50,60,110],
    [70,80,90,120]
])

print(x[0])
print(x[-1])
print(x[:,0])
print(x[:,-1])
"""

#Exercise 3
"""
x = torch.arange(36)
x = x.reshape(6,6)
print(x)
print(x[2:4,2:4])
"""

#Exercise 4
"""
x = torch.randn(2,3,4)
print(x.shape)
print(x.ndim)
print(x.reshape(6,4))
"""

#Exercise 5
"""
x = torch.tensor([[1,2,3,4,5],
                 [1,2,3,4,5],
                 [1,2,3,4,5],
                 [1,2,3,4,5]])
#print(x.unsqueeze(2))
#print(x.squeeze(1))
"""