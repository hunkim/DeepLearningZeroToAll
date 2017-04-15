# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# Video: https://www.youtube.com/watch?v=ftMq5ps503w
import numpy as np
import mxnet as mx
import logging
import sys
from sklearn.preprocessing import MinMaxScaler

# brew install graphviz
# pip3 install graphviz
# pip3 install pydot-ng
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Config the logging
np.random.seed(777)
mx.random.seed(777)

timesteps = seq_length = 7
batch_size = 32
data_dim = 5


def build_sym(seq_len, use_cudnn=False):
    """Build the symbol for stock-price prediction

    Parameters
    ----------
    seq_len : int
    use_cudnn : bool, optional
        Whether to use the LSTM implemented in cudnn, will be faster than the original version

    Returns
    -------
    pred : mx.sym.Symbol
        The prediction result
    """
    data = mx.sym.var("data")  # Shape: (N, T, C)
    target = mx.sym.var("target")  # Shape: (N, T, C)
    data = mx.sym.transpose(data, axes=(1, 0, 2))  # Shape: (T, N, C)
    if use_cudnn:
        lstm1 = mx.rnn.FusedRNNCell(num_hidden=5, mode="lstm", prefix="lstm1_")
        lstm2 = mx.rnn.FusedRNNCell(num_hidden=10, mode="lstm", prefix="lstm2_",
                                    get_next_state=True)
    else:
        lstm1 = mx.rnn.LSTMCell(num_hidden=5, prefix="lstm1_")
        lstm2 = mx.rnn.LSTMCell(num_hidden=10, prefix="lstm2_")
    L1, _ = lstm1.unroll(length=seq_len, inputs=data, merge_outputs=True,
                         layout="TNC")  # Shape: (T, N, 5)
    L1 = mx.sym.Dropout(L1, p=0.2)  # Shape: (T, N, 5)
    _, L2_states = lstm2.unroll(length=seq_len, inputs=L1, merge_outputs=True,
                                layout="TNC")  # Shape: (T, N, 10)
    L2 = mx.sym.reshape(L2_states[0], shape=(-1, 0), reverse=True)  # Shape: (T * N, 10)
    pred = mx.sym.FullyConnected(L2, num_hidden=1, name="pred")
    pred = mx.sym.LinearRegressionOutput(data=pred, label=target)
    return pred

# Open,High,Low,Close,Volume
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)

# very important. It does not work without it.
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)

x = xy
y = xy[:, [-1]]  # Close as label

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# split to train and testing
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])


def train_eval_net(use_cudnn):
    pred = build_sym(seq_len=seq_length, use_cudnn=use_cudnn)
    net = mx.mod.Module(symbol=pred, data_names=['data'], label_names=['target'], context=mx.gpu())

    train_iter = mx.io.NDArrayIter(data=trainX, label=trainY,
                                   data_name="data", label_name="target",
                                   batch_size=batch_size,
                                   shuffle=True)
    test_iter = mx.io.NDArrayIter(data=testX, label=testY,
                                  data_name="data", label_name="target",
                                  batch_size=batch_size)
    net.fit(train_data=train_iter, eval_data=test_iter,
            initializer=mx.init.Xavier(rnd_type="gaussian", magnitude=1),
            optimizer="adam",
            optimizer_params={"learning_rate": 1E-3},
            eval_metric="mse", num_epoch=200)

    # make predictions
    testPredict = net.predict(test_iter).asnumpy()
    mse = np.mean((testPredict - testY)**2)
    return testPredict, mse

# inverse values
# testPredict = scaler.transform(testPredict)
# testY = scaler.transform(testY)
import time
print("Begin to train LSTM with CUDNN acceleration...")
begin = time.time()
cudnn_pred, cudnn_mse = train_eval_net(use_cudnn=True)
end = time.time()
cudnn_time_spent = end - begin
print("Done!")

print("Begin to train LSTM without CUDNN acceleration...")
begin = time.time()
normal_pred, normal_mse = train_eval_net(use_cudnn=False)
end = time.time()
normal_time_spent = end - begin
print("Done!")


print("CUDNN time spent: %g, test mse: %g" % (cudnn_time_spent, cudnn_mse))
print("NoCUDNN time spent: %g, test mse: %g" % (normal_time_spent, normal_mse))

plt.close('all')
fig = plt.figure()
plt.plot(testY, label='Groud Truth')
plt.plot(cudnn_pred, label='With cuDNN')
plt.plot(normal_pred, label='Without cuDNN')
plt.legend()
plt.show()
'''
CUDNN time spent: 10.0955, test mse: 0.00721571
NoCUDNN time spent: 38.9882, test mse: 0.00565724
'''
