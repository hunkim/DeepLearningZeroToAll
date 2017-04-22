# Lab 12 Character Sequence RNN
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

# hyperparameters
learning_rate = 0.1
num_epochs = 500
input_size = len(char_set)  # RNN input size (one hot size)
hidden_size = len(char_set)  # RNN output size
num_classes = len(char_set)  # final output size (RNN or softmax, etc.)
sequence_length = 10  # any arbitrary number
num_layers = 2  # number of layers in RNN

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

x_data = torch.Tensor(dataX)
y_data = torch.LongTensor(dataY)

# one hot encoding


def one_hot(x, num_classes):
    idx = x.long()
    idx = idx.view(-1, 1)
    x_one_hot = torch.zeros(x.size()[0] * x.size()[1], num_classes)
    x_one_hot.scatter_(1, idx, 1)
    x_one_hot = x_one_hot.view(x.size()[0], x.size()[1], num_classes)
    return x_one_hot


x_one_hot = one_hot(x_data, num_classes)

inputs = Variable(x_one_hot)
labels = Variable(y_data)


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        # Set parameters for RNN block
        # Note: batch_first=False by default.
        # When true, inputs are (batch_size, sequence_length, input_dimension)
        # instead of (sequence_length, batch_size, input_dimension)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # h_0 = Variable(torch.zeros(
        # self.num_layers, x.size(0), self.hidden_size))
        # c_0 = Variable(torch.zeros(
        # self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        # Input: (batch, seq_len, input_size)
        out, _ = self.lstm(x, (h_0, c_0))
        # Note: the output tensor of LSTM in this case is a block with holes
        # > add .contiguous() to apply view()
        out = out.contiguous().view(-1, self.hidden_size)
        # Return outputs applied to fully connected layer
        out = self.fc(out)
        return out


# Instantiate RNN model
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

# Set loss and optimizer function
criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(inputs)
    optimizer.zero_grad()
    # obtain the loss function
    # flatten target labels to match output
    loss = criterion(outputs, labels.view(-1))
    loss.backward()
    optimizer.step()
    # obtain the predicted indices of the next character
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    idx = idx.reshape(-1, sequence_length)  # (170,10)
    # display the prediction of the last sequence
    result_str = [char_set[c] for c in idx[-1]]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")
