#!/usr/bin/env python
# Lab 2-1 Linear Regression

import numpy as np
import chainer
from chainer import training
from chainer import datasets
from chainer.training import extensions

import chainer.functions as F
import chainer.links as L


class MyModel(chainer.Chain):
    # Define model to be called later by L.Classifier()

    def __init__(self, n_out):
        super(MyModel, self).__init__(
            l1=L.Linear(None, n_out),
        )

    def __call__(self, x):
        return self.l1(x)


def generate_data():
    # Need to reshape so that each input is an array.
    reshape = lambda x: np.reshape(x, (len(x), 1))

    # Notice the type specification (np.float32)
    # For regression, use np.float32 for both input & output, while for
    # classification using softmax_cross_entropy, the output(label) needs to be
    # of type np.int32.
    X = np.linspace(-1, 1, 101).astype(np.float32)
    Y = (2 * X + np.random.randn(*X.shape) * 0.33).astype(np.float32)
    return reshape(X), reshape(Y)


def main():
    epoch = 100
    batch_size = 1

    data = generate_data()

    # Convert to set of tuples (target, label).
    train = datasets.TupleDataset(*data)

    model = L.Classifier(MyModel(1), lossfun=F.mean_squared_error)

    # Set compute_accuracy=False when using MSE.
    model.compute_accuracy = False

    # Define optimizer (Adam, RMSProp, etc)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Define iterators.
    train_iter = chainer.iterators.SerialIterator(train, batch_size)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    # Helper functions (extensions) to monitor progress on stdout.
    report_params = [
        'epoch',
        'main/loss',
    ]
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())

    # Run trainer
    trainer.run()

    # Should print out value close to 2.
    print(model.predictor(np.array([[1]]).astype(np.float32)).data)

if __name__ == "__main__":
    main()


"""
Expected output.
---

epoch       main/loss
...
90          0.104054
91          0.104079
92          0.104037
93          0.104005
94          0.104142
95          0.104292
96          0.103934
97          0.104091
98          0.103952
99          0.104034
100         0.103947
[[ 1.98888695]]
"""
