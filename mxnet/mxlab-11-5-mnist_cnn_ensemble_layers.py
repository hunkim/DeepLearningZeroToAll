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
training_epochs = 20
batch_size = 100
num_models = 2


def build_symbol():
    data = mx.sym.var(name="data")
    label = mx.sym.var(name="label")

    L1 = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=32, name='L1_conv')
    L1 = mx.sym.Activation(data=L1, act_type='relu', name='L1_relu')
    L1 = mx.sym.Pooling(data=L1, kernel=(2, 2), stride=(2, 2), pool_type='max', name='L1_pool')
    L1 = mx.sym.Dropout(L1, p=0.3, name="L1_dropout")

    L2 = mx.sym.Convolution(data=L1, kernel=(3, 3), pad=(1, 1), num_filter=64, name='L2_conv')
    L2 = mx.sym.Activation(data=L2, act_type='relu', name='L2_relu')
    L2 = mx.sym.Pooling(data=L2, kernel=(2, 2), stride=(2, 2), pool_type='max', name='L2_pool')
    L2 = mx.sym.Dropout(L2, p=0.3, name="L2_dropout")

    L3 = mx.sym.Convolution(data=L2, kernel=(3, 3), pad=(1, 1), num_filter=128, name='L3_conv')
    L3 = mx.sym.Activation(data=L3, act_type='relu', name='L3_relu')
    L3 = mx.sym.Pooling(data=L3, kernel=(2, 2), stride=(2, 2), pad=(1, 1), pool_type='max', name='L3_pool')
    L3 = mx.sym.flatten(L3)
    L3 = mx.sym.Dropout(L3, p=0.3, name="L3_dropout")

    L4 = mx.sym.FullyConnected(data=L3, num_hidden=625, name='L4_fc')
    L4 = mx.sym.Dropout(L4, p=0.5)

    logits = mx.sym.FullyConnected(data=L4, num_hidden=10, name='logits')

    loss = mx.sym.mean(-mx.sym.pick(mx.sym.log_softmax(logits), label, axis=-1))
    loss = mx.sym.make_loss(loss)
    return loss, logits


def get_batch(p, batch_size, X, y):
    data_npy = np.take(X,
                       indices=np.arange(p * batch_size, (p + 1) * batch_size),
                       axis=0,
                       mode="clip")
    label_npy = np.take(y,
                        indices=np.arange(p * batch_size, (p + 1) * batch_size),
                        axis=0,
                        mode="clip")
    num_valid = batch_size if (p + 1) * batch_size <= X.shape[0] else X.shape[0] - p * batch_size
    return data_npy, label_npy, num_valid


train_nets = []
test_nets = []
# 1. Get the symbol
loss, logits = build_symbol()

# 2. Build the training nets and testing nets
data_desc = mx.io.DataDesc(name='data', shape=(batch_size, 1, 28, 28), layout='NCHW')
label_desc = mx.io.DataDesc(name='label', shape=(batch_size, ), layout='N')
for i in range(num_models):
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
    train_nets.append(net)
    test_nets.append(test_net)

print('Learning Started!')

# 3. Train all the models
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(num_models)
    total_batch = int(math.ceil(X_train.shape[0] / batch_size))
    shuffle_ind = np.random.permutation(np.arange(X_train.shape[0]))
    X_train = X_train[shuffle_ind, :]
    y_train = y_train[shuffle_ind]
    for i in range(total_batch):
        data_npy, label_npy, _ = get_batch(i, batch_size, X_train, y_train)
        for i, net in enumerate(train_nets):
            net.forward(data_batch=mx.io.DataBatch(data=[nd.array(data_npy)],
                                                   label=[nd.array(label_npy)]),
                        is_train=True)
            loss_nd = net.get_outputs()[0]
            net.backward()
            net.update()
            avg_cost_list[i] += loss_nd.asnumpy()[0] / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')

# 5. Test the networks
total_batch = int(np.ceil(X_test.shape[0] / batch_size))
correct_counts = [0 for i in range(num_models)]
ensemble_correct_count = 0
total_num = 0
for i in range(total_batch):
    num_valid = batch_size if (i + 1) * batch_size <= X_test.shape[0]\
        else X_test.shape[0] - i * batch_size
    data_npy, label_npy, num_valid = get_batch(i, batch_size, X_test, y_test)
    prob_ensemble = nd.zeros(shape=(label_npy.shape[0], 10), ctx=mx.gpu())
    for i, test_net in enumerate(test_nets):
        test_net.forward(data_batch=mx.io.DataBatch(data=[nd.array(data_npy)],
                                                    label=None),
                         is_train=False)
        logits_nd = test_net.get_outputs()[0]
        prob_nd = nd.softmax(logits_nd)
        prob_ensemble += prob_nd
        pred_cls = nd.argmax(prob_nd, axis=-1).asnumpy()
        correct_counts[i] += (pred_cls[:num_valid] == label_npy[:num_valid]).sum()
    prob_ensemble /= num_models
    ensemble_pred_cls = nd.argmax(prob_ensemble, axis=-1).asnumpy()
    ensemble_correct_count += (ensemble_pred_cls[:num_valid] == label_npy[:num_valid]).sum()
for i in range(num_models):
    print(i, 'Accuracy:', correct_counts[i] / float(X_test.shape[0]))
print('Ensemble accuracy:', ensemble_correct_count / float(X_test.shape[0]))
'''
Learning Started!
Epoch: 0001 cost = [ 0.23813407  0.23717315]
Epoch: 0002 cost = [ 0.07455271  0.07434764]
Epoch: 0003 cost = [ 0.05925059  0.06024327]
Epoch: 0004 cost = [ 0.05032205  0.04895757]
Epoch: 0005 cost = [ 0.04573197  0.0439943 ]
Epoch: 0006 cost = [ 0.04143022  0.0416003 ]
Epoch: 0007 cost = [ 0.03805082  0.03796253]
Epoch: 0008 cost = [ 0.03668946  0.03679928]
Epoch: 0009 cost = [ 0.03688032  0.03588339]
Epoch: 0010 cost = [ 0.03180911  0.03446447]
Epoch: 0011 cost = [ 0.03293695  0.03334761]
Epoch: 0012 cost = [ 0.03255253  0.03101865]
Epoch: 0013 cost = [ 0.03044157  0.03092092]
Epoch: 0014 cost = [ 0.02833735  0.02996393]
Epoch: 0015 cost = [ 0.03077925  0.02817958]
Epoch: 0016 cost = [ 0.02788243  0.02807305]
Epoch: 0017 cost = [ 0.02659359  0.02851706]
Epoch: 0018 cost = [ 0.02834567  0.02659114]
Epoch: 0019 cost = [ 0.02670252  0.02910407]
Epoch: 0020 cost = [ 0.02647125  0.02396415]
Learning Finished!
0 Accuracy: 0.9924
1 Accuracy: 0.9912
Ensemble accuracy: 0.9938
'''
