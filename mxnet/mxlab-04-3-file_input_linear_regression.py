# Lab 4 Linear Regression
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Config the logging
np.random.seed(777)

# 1. Prepare Data
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print("x_data", x_data)
print("y_data", y_data)

# hyper-parameters
sample_num = x_data.shape[0]
dimension = x_data.shape[1]
batch_size = 25

# 2. Build the Linear Regression Symbol
data = mx.sym.Variable("data")
target = mx.sym.Variable("target")
fc = mx.sym.FullyConnected(data=data, num_hidden=1, name='fc')
pred = mx.sym.LinearRegressionOutput(data=fc, label=target)

# 3. Construct the Module based on the symbol.
net = mx.mod.Module(symbol=pred,
                    data_names=['data'],
                    label_names=['target'],
                    context=mx.gpu(0))
net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(batch_size, dimension), layout='NC')],
         label_shapes=[mx.io.DataDesc(name='target', shape=(batch_size, 1), layout='NC')])
net.init_params(initializer=mx.init.Normal(sigma=0.01))
net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 1E-4, 'momentum': 0.9})

# 4. Train the model
# First constructing the training iterator and then fit the model
train_iter = mx.io.NDArrayIter(x_data, y_data, batch_size, shuffle=True, label_name='target')
net.fit(train_data=train_iter, eval_metric="mse", num_epoch=2000)

# 5. Test the model
test_net = mx.mod.Module(symbol=fc,
                         data_names=['data'],
                         label_names=None,
                         context=mx.gpu(0))
test_net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(1, dimension), layout='NC')],
              label_shapes=None,
              for_training=False,
              shared_module=net)
test_net.forward(mx.io.DataBatch(data=[nd.array([[60, 70, 110]])], label=None))
print("input = [60, 70, 110], score =", test_net.get_outputs()[0].asnumpy())
test_net.forward(mx.io.DataBatch(data=[nd.array([[90, 100, 80]])], label=None))
print("input = [90, 100, 80], score =", test_net.get_outputs()[0].asnumpy())
'''
input = [60, 70, 110], score = [[ 182.48858643]]
input = [90, 100, 80], score = [[ 175.24279785]]
'''
