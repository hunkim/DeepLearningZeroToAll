# Lab 11 MNIST and Deep learning CNN
import math
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import random
from sklearn.datasets import fetch_mldata

# set the seeds. However, this does not guarantee that the result will always be the same since CUDNN is non-deterministic
np.random.seed(777)
mx.random.seed(77)
random.seed(7777)

# 1. Loading MNIST
mnist = fetch_mldata(dataname='MNIST original')
X, y = mnist.data, mnist.target
X = X.astype(np.float32) / 255.0
X_train, X_valid, X_test = X[:55000].reshape((-1, 1, 28, 28)),\
    X[55000:60000].reshape((-1, 1, 28, 28)),\
    X[60000:].reshape((-1, 1, 28, 28))
y_train, y_valid, y_test = y[:55000], y[55000:60000], y[60000:]

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
drop_out_prob = 0.3  # The keep probability is 0.7

# 2. Build symbol
data = mx.sym.var(name="data")
label = mx.sym.var(name="label")

L1 = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=32, name='L1_conv')
L1 = mx.sym.Activation(data=L1, act_type='relu', name='L1_relu')
L1 = mx.sym.Pooling(data=L1, kernel=(2, 2), stride=(2, 2), pool_type='max', name='L1_pool')
L1 = mx.sym.Dropout(L1, p=drop_out_prob, name="L1_dropout")

L2 = mx.sym.Convolution(data=L1, kernel=(3, 3), pad=(1, 1), num_filter=64, name='L2_conv')
L2 = mx.sym.Activation(data=L2, act_type='relu', name='L2_relu')
L2 = mx.sym.Pooling(data=L2, kernel=(2, 2), stride=(2, 2), pool_type='max', name='L2_pool')
L2 = mx.sym.Dropout(L2, p=drop_out_prob, name="L2_dropout")

L3 = mx.sym.Convolution(data=L2, kernel=(3, 3), pad=(1, 1), num_filter=128, name='L3_conv')
L3 = mx.sym.Activation(data=L3, act_type='relu', name='L3_relu')
L3 = mx.sym.Pooling(data=L3, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='max', name='L3_pool')
L3 = mx.sym.flatten(L3)
L3 = mx.sym.Dropout(L3, p=drop_out_prob, name="L3_dropout")

L4 = mx.sym.FullyConnected(data=L3, num_hidden=625, name='L4_fc')
L4 = mx.sym.Dropout(L4, p=drop_out_prob)

logits = mx.sym.FullyConnected(data=L4, num_hidden=10, name='logits')

loss = mx.sym.mean(-mx.sym.pick(mx.sym.log_softmax(logits), label, axis=-1))
loss = mx.sym.make_loss(loss)

# 3. Build network handler
data_desc = mx.io.DataDesc(name='data', shape=(batch_size, 1, 28, 28), layout='NCHW')
label_desc = mx.io.DataDesc(name='label', shape=(batch_size, ), layout='N')
net = mx.mod.Module(symbol=loss,
                    data_names=[data_desc.name],
                    label_names=[label_desc.name],
                    context=mx.gpu())
net.bind(data_shapes=[data_desc], label_shapes=[label_desc])
net.init_params(initializer=mx.init.Xavier())
net.init_optimizer(optimizer="adam",
                   optimizer_params={'learning_rate': learning_rate,
                                     'rescale_grad': 1.0},
                   kvstore=None)

# We build another testing network that outputs the logits.
test_net = mx.mod.Module(symbol=logits,
                         data_names=[data_desc.name],
                         label_names=None,
                         context=mx.gpu())
# Setting the `shared_module` to ensure that the test network shares the same parameters and
#  allocated memory of the training network
test_net.bind(data_shapes=[data_desc],
              label_shapes=None,
              for_training=False,
              grad_req='null',
              shared_module=net)

# 4. Train the network
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(math.ceil(X_train.shape[0] / batch_size))
    shuffle_ind = np.random.permutation(np.arange(X_train.shape[0]))
    X_train = X_train[shuffle_ind, :]
    y_train = y_train[shuffle_ind]
    for i in range(total_batch):
        # Slice the data batch and label batch.
        # Note that we use np.take to ensure that the batch will be padded correctly.
        data_npy = np.take(X_train,
                           indices=np.arange(i * batch_size, (i + 1) * batch_size),
                           axis=0,
                           mode="clip")
        label_npy = np.take(y_train,
                            indices=np.arange(i * batch_size, (i + 1) * batch_size),
                            axis=0,
                            mode="clip")
        net.forward(data_batch=mx.io.DataBatch(data=[nd.array(data_npy)],
                                               label=[nd.array(label_npy)]),
                    is_train=True)
        loss_nd = net.get_outputs()[0]
        net.backward()
        net.update()
        avg_cost += loss_nd.asnumpy()[0] / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')

# 5. Test the network
total_batch = int(np.ceil(X_test.shape[0] / batch_size))
correct_count = 0
total_num = 0
for i in range(total_batch):
    num_valid = batch_size if (i + 1) * batch_size <= X_test.shape[0]\
        else X_test.shape[0] - i * batch_size
    data_npy = np.take(X_test,
                       indices=np.arange(i * batch_size, (i + 1) * batch_size),
                       axis=0,
                       mode="clip")
    label_npy = np.take(y_test,
                        indices=np.arange(i * batch_size, (i + 1) * batch_size),
                        axis=0,
                        mode="clip")
    test_net.forward(data_batch=mx.io.DataBatch(data=[nd.array(data_npy)],
                                                label=None),
                     is_train=False)
    logits_nd = test_net.get_outputs()[0]
    pred_cls = nd.argmax(logits_nd, axis=-1).asnumpy()
    correct_count += (pred_cls[:num_valid] == label_npy[:num_valid]).sum()
acc = correct_count / float(X_test.shape[0])
print('Accuracy:', acc)

# 6. Get one and predict
test_net.reshape(data_shapes=[mx.io.DataDesc(name='data', shape=(1, 1, 28, 28), layout='NCHW')],
                 label_shapes=None)
r = np.random.randint(0, X_test.shape[0])
test_net.forward(data_batch=mx.io.DataBatch(data=[nd.array(X_test[r:r + 1])],
                                            label=None))
logits_nd = test_net.get_outputs()[0]
print("Label: ", int(y_test[r]))
print("Prediction: ", int(nd.argmax(logits_nd, axis=1).asnumpy()[0]))
'''
Epoch: 0001 cost = 0.222577997
Epoch: 0002 cost = 0.072177568
Epoch: 0003 cost = 0.055896563
Epoch: 0004 cost = 0.049281721
Epoch: 0005 cost = 0.042741676
Epoch: 0006 cost = 0.040903398
Epoch: 0007 cost = 0.037740686
Epoch: 0008 cost = 0.035468433
Epoch: 0009 cost = 0.035284971
Epoch: 0010 cost = 0.032539701
Epoch: 0011 cost = 0.031544954
Epoch: 0012 cost = 0.032199799
Epoch: 0013 cost = 0.030421943
Epoch: 0014 cost = 0.028435153
Epoch: 0015 cost = 0.028335706
Learning Finished!
Accuracy: 0.989
Label:  2
Prediction:  2
'''
