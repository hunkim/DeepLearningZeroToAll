
# coding: utf-8

# In[1]:

import torch
from torch.autograd import Variable 
import numpy as np

torch.manual_seed(777)

x_data = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

print(X,Y)

# Our model
linear = torch.nn.Linear(2,1,bias=True)
sigmoid = torch.nn.Sigmoid()

model = torch.nn.Sequential(linear,sigmoid)
# model:add(linear)
# model:add(sigmoid)

# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)
#     cost = criterion(hypothesis,Y)
    cost = -(Y*torch.log(hypothesis) + (1-Y)*torch.log(1-hypothesis)).mean()
    cost.backward()
    optimizer.step()
    
    if step % 200 == 0:
        print(step, cost.data.numpy())
        
    
predicted = (model(X).data > 0.5).float()
accuracy = (predicted == Y.data).float().mean()
print("\nHypothesis: ", hypothesis.data.numpy(), "\nCorrect (Y): ", predicted.numpy(), "\nAccuracy: ", accuracy)

