# Lab 12 Character Sequence RNN
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility

sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> index

# hyperparameters
learning_rate = 0.1
num_epochs = 50
input_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
num_layers = 1  # number of layers in RNN

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

x_data = torch.Tensor(x_data)
y_data = torch.LongTensor(y_data)

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
        # Fully connected layer to obtain outputs corresponding to the number
        # of classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Reshape input
        x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))

        # Reshape output from (batch, seq_len, hidden_size) to (batch *
        # seq_len, hidden_size)
        out = out.view(-1, self.hidden_size)
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
    loss = criterion(outputs, labels.view(-1))
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")
