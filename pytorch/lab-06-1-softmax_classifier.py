# Lab 6 Softmax Classifier
import torch
from torch.autograd import Variable

torch.manual_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = Variable(torch.Tensor(x_data))
Y = Variable(torch.Tensor(y_data))
nb_classes = 3

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
softmax = torch.nn.Softmax()
linear = torch.nn.Linear(4, nb_classes, bias=True)
model = torch.nn.Sequential(linear, softmax)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    # Cross entropy cost/loss
    cost = -Y * torch.log(hypothesis)
    cost = torch.sum(cost, 1).mean()
    cost.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, cost.data.numpy())

# Testing & One-hot encoding
print('--------------')
a = model(Variable(torch.Tensor([[1, 11, 7, 9]])))
print(a.data.numpy(), torch.max(a, 1)[1].data.numpy())

print('--------------')
b = model(Variable(torch.Tensor([[1, 3, 4, 3]])))
print(b.data.numpy(), torch.max(b, 1)[1].data.numpy())

print('--------------')
c = model(Variable(torch.Tensor([[1, 1, 0, 1]])))
print(c.data.numpy(), torch.max(c, 1)[1].data.numpy())

print('--------------')
all = model(Variable(torch.Tensor([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]])))
print(all.data.numpy(), torch.max(all, 1)[1].data.numpy())
