import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

np.random.seed(777)

# 1. Prepare Data
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print("x_data", x_data)
print("y_data", y_data)

sample_num = x_data.shape[0]
dimension = x_data.shape[1]
batch_size = 25
train_iter = mx.io.NDArrayIter(x_data, y_data, batch_size, shuffle=True, label_name='lin_reg_label')

# 2. Build Linear Regression Model
data = mx.sym.Variable("data")
target = mx.sym.Variable("target")
pred = mx.sym.FullyConnected(data=data, num_hidden=1, name='pred')
loss = mx.sym.mean(mx.sym.square(target - pred))
loss = mx.sym.make_loss(loss)


# 3. Build the network
net = mx.mod.Module(symbol=loss,
                    data_names=['data'],
                    label_names=['target'],
                    context=mx.gpu(0))
net.bind(data_shapes = [mx.io.DataDesc(name='data', shape=(batch_size, 3), layout='NC')],
         label_shapes= [mx.io.DataDesc(name='target', shape=(batch_size, 1), layout='NC')])
net.init_params(initializer=mx.init.Normal(sigma=0.01))
net.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 1E-3, 'momentum':0.9})

test_net = mx.mod.Module(symbol=pred,
                         data_names=['data'],
                         label_names=None,
                         context=mx.gpu(0))
test_net.bind(data_shapes=[mx.io.DataDesc(name='data', shape=(sample_num, 3), layout='NC')],
              label_shapes=None,
              shared_module=net)

# 4. Train the model
y_data_nd = nd.array(y_data, ctx=mx.gpu())
for idx in range(2000):
    total_batch_num = int(sample_num / batch_size)
    for batch_id in range(total_batch_num):
        x_batch = x_data[batch_id*batch_size:(batch_id+1)*batch_size , :]
        y_batch = y_data[batch_id*batch_size:(batch_id+1)*batch_size , :]
        data_batch = mx.io.DataBatch(data=[mx.nd.array(x_batch)],
                                     label=[mx.nd.array(y_batch)])
        net.forward_backward(data_batch)
        net.update()
    test_net.forward(mx.io.DataBatch(data=[mx.nd.array(x_data)], label=None), is_train=False)
    pred_nd = test_net.get_outputs()[0]
    mse = nd.mean(nd.square(pred_nd - y_data_nd)).asnumpy()[0]
    print("epoch: %d, mse = %g" %(idx, mse))

test_net.reshape(data_shapes=[["data", (1, 3)]])
test_net.forward(mx.io.DataBatch(data=[nd.array([[60, 70, 110]])], label=None))
print("input = [60, 70, 110], score =", test_net.get_outputs()[0].asnumpy())
test_net.forward(mx.io.DataBatch(data=[nd.array([[90, 100, 80]])], label=None))
print("input = [90, 100, 80], score =", test_net.get_outputs()[0].asnumpy())
'''
input = [60, 70, 110], score = [[ 182.48858643]]
input = [90, 100, 80], score = [[ 175.24279785]]
'''