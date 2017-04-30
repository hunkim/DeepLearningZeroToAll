# Lab 5 Logistic Regression Classifier
import mxnet as mx
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Config the logging
np.random.seed(777)

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# hyper-parameters
sample_num = x_data.shape[0]
dimension = x_data.shape[1]
batch_size = 32

# 2. Build the Logistic Regression Symbol
data = mx.sym.Variable("data")
target = mx.sym.Variable("target")
fc = mx.sym.FullyConnected(data=data, num_hidden=1, name='fc')
pred = mx.sym.LogisticRegressionOutput(data=fc, label=target)

# 3. Construct the Module based on the symbol.
net = mx.mod.Module(symbol=pred,
                    data_names=['data'],
                    label_names=['target'],
                    context=mx.gpu(0))
net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(batch_size, dimension), layout='NC')],
         label_shapes=[mx.io.DataDesc(name='target', shape=(batch_size, 1), layout='NC')])
net.init_params(initializer=mx.init.Normal(sigma=0.01))
net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 1E-3, 'momentum': 0.9})

# 4. Train the model
# First constructing the training iterator and then fit the model
train_iter = mx.io.NDArrayIter(x_data, y_data, batch_size, shuffle=True, label_name='target')
metric = mx.metric.CustomMetric(feval=lambda labels, pred: ((pred > 0.5) == labels).mean(),
                                name="acc")
net.fit(train_data=train_iter, eval_metric=metric, num_epoch=200)

'''
...
INFO:root:Epoch[195] Train-acc=0.770833
INFO:root:Epoch[195] Time cost=0.015
INFO:root:Epoch[196] Train-acc=0.770833
INFO:root:Epoch[196] Time cost=0.015
INFO:root:Epoch[197] Train-acc=0.770833
INFO:root:Epoch[197] Time cost=0.013
INFO:root:Epoch[198] Train-acc=0.769531
INFO:root:Epoch[198] Time cost=0.012
INFO:root:Epoch[199] Train-acc=0.769531
INFO:root:Epoch[199] Time cost=0.012
'''

# 5. Test the model
test_iter = mx.io.NDArrayIter(x_data, None, batch_size, shuffle=False, label_name=None)

pred_class = (fc > 0)
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
...
[ 1.]
[ 1.]
[ 1.]
[ 0.]
[ 1.]
[ 0.]
[ 1.]
[ 1.]
[ 1.]
[ 1.]
[ 1.]
[ 1.]]
'''
