# Lab 6 Softmax Classifier
import torch
from torch.autograd import Variable
import numpy as np

torch.manual_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7  # 0 ~ 6

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

# one hot encoding
Y_one_hot = torch.zeros(Y.size()[0], nb_classes)
Y_one_hot.scatter_(1, Y.long().data, 1)
Y_one_hot = Variable(Y_one_hot)
print("one_hot", Y_one_hot.data)

softmax = torch.nn.Softmax()
model = torch.nn.Linear(16, nb_classes, bias=True)

# Cross entropy cost/loss
criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    # Label has to be 1D LongTensor
    cost = criterion(hypothesis, Y.long().view(-1))
    cost.backward()
    optimizer.step()

    prediction = torch.max(softmax(hypothesis), 1)[1].float()

    correct_prediction = (prediction.data == Y.data)
    accuracy = correct_prediction.float().mean()

    if step % 100 == 0:
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, cost.data[0], accuracy))


# Let's see if we can predict
pred = torch.max(softmax(hypothesis), 1)[1].float()

for p, y in zip(pred, Y):
    print("[{}] Prediction: {} True Y: {}".format(bool(p.data[0] == y.data[0]), p.data.int()[0], y.data.int()[0]))
