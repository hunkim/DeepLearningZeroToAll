# Lab 4 Multi-variable linear regression
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)   # for reproducibility

# X and Y data
x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

X = Variable(torch.Tensor(x_data))
Y = Variable(torch.Tensor(y_data))

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
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step, "Cost: ", cost.data.numpy(), "\nPrediction:\n", hypothesis.data.numpy())
