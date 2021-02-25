# Lab 6 Softmax Classifier
import mxnet as mx
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Config the logging
np.random.seed(777)

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]].reshape((-1,))

print(x_data.shape, y_data.shape)

# hyper-parameters
nb_classes = 7  # 0 ~ 6
sample_num = x_data.shape[0]
dimension = x_data.shape[1]
batch_size = 32

# 2. Build the Softmax Classification Symbol
data = mx.sym.Variable("data")
target = mx.sym.Variable("target")
logits = mx.sym.FullyConnected(data=data, num_hidden=nb_classes, name='logits')
pred = mx.sym.SoftmaxOutput(data=logits, label=target)

# 3. Construct the Module based on the symbol.
net = mx.mod.Module(symbol=pred,
                    data_names=['data'],
                    label_names=['target'],
                    context=mx.gpu(0))
net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(batch_size, dimension), layout='NC')],
         label_shapes=[mx.io.DataDesc(name='target', shape=(batch_size,), layout='NC')])
net.init_params(initializer=mx.init.Normal(sigma=0.01))
net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 1E-1, 'momentum': 0.9})

# 4. Train the model
# First constructing the training iterator and then fit the model
train_iter = mx.io.NDArrayIter(x_data, y_data, batch_size, shuffle=True, label_name='target')
net.fit(train_data=train_iter, eval_metric='acc', num_epoch=40)

'''
INFO:root:Epoch[27] Train-accuracy=0.992188
INFO:root:Epoch[27] Time cost=0.003
INFO:root:Epoch[28] Train-accuracy=0.992188
INFO:root:Epoch[28] Time cost=0.003
INFO:root:Epoch[29] Train-accuracy=0.992188
INFO:root:Epoch[29] Time cost=0.003
INFO:root:Epoch[30] Train-accuracy=1.000000
INFO:root:Epoch[30] Time cost=0.003
INFO:root:Epoch[31] Train-accuracy=1.000000
INFO:root:Epoch[31] Time cost=0.003
INFO:root:Epoch[32] Train-accuracy=1.000000
INFO:root:Epoch[32] Time cost=0.003
INFO:root:Epoch[33] Train-accuracy=1.000000
INFO:root:Epoch[33] Time cost=0.004
INFO:root:Epoch[34] Train-accuracy=1.000000
INFO:root:Epoch[34] Time cost=0.003
INFO:root:Epoch[35] Train-accuracy=1.000000
INFO:root:Epoch[35] Time cost=0.003
INFO:root:Epoch[36] Train-accuracy=1.000000
INFO:root:Epoch[36] Time cost=0.003
INFO:root:Epoch[37] Train-accuracy=1.000000
INFO:root:Epoch[37] Time cost=0.004
INFO:root:Epoch[38] Train-accuracy=1.000000
INFO:root:Epoch[38] Time cost=0.003
INFO:root:Epoch[39] Train-accuracy=1.000000
INFO:root:Epoch[39] Time cost=0.003
'''
# 5. Test the model
test_iter = mx.io.NDArrayIter(x_data, None, batch_size, shuffle=False, label_name=None)

pred_class = mx.sym.argmax(logits, axis=-1, name="pred_class")
test_net = mx.mod.Module(symbol=pred_class,
                         data_names=['data'],
                         label_names=None,
                         context=mx.gpu(0))
test_net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(batch_size, dimension), layout='NC')],
              label_shapes=None,
              for_training=False,
              shared_module=net)
out = test_net.predict(eval_data=test_iter)
print(out.asnumpy())
'''
[ 0.  0.  3.  0.  0.  0.  0.  3.  3.  0.  0.  1.  3.  6.  6.  6.  1.  0.
  3.  0.  1.  1.  0.  1.  5.  4.  4.  0.  0.  0.  5.  0.  0.  1.  3.  0.
  0.  1.  3.  5.  5.  1.  5.  1.  0.  0.  6.  0.  0.  0.  0.  5.  4.  6.
  0.  0.  1.  1.  1.  1.  3.  3.  2.  0.  0.  0.  0.  0.  0.  0.  0.  1.
  6.  3.  0.  0.  2.  6.  1.  1.  2.  6.  3.  1.  0.  6.  3.  1.  5.  4.
  2.  2.  3.  0.  0.  1.  0.  5.  0.  6.  1.]
'''
