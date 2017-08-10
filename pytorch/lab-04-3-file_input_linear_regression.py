# Lab 4 Multi-variable linear regression
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

torch.manual_seed(777)   # for reproducibility

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

x_data = Variable(torch.from_numpy(x_data))
y_data = Variable(torch.from_numpy(y_data))

# Our hypothesis XW+b
model = nn.Linear(3, 1, bias=True)

# cost criterion
criterion = nn.MSELoss()

# Minimize
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

# Train the model
for step in range(2001):
    optimizer.zero_grad()
    # Our hypothesis
    hypothesis = model(x_data)
    cost = criterion(hypothesis, y_data)
    cost.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step, "Cost: ", cost.data.numpy(), "\nPrediction:\n", hypothesis.data.numpy())

# Ask my score
print("Your score will be ", model(Variable(torch.Tensor([[100, 70, 101]]))).data.numpy())
print("Other scores will be ", model(Variable(torch.Tensor([[60, 70, 110], [90, 100, 80]]))).data.numpy())
