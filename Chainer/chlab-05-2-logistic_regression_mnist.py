#!/usr/bin/env python
# Lab 5 Logistic Regression Classifier

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions


class MyModel(chainer.Chain):
    # Define model to be called later by L.Classifer()

    def __init__(self, n_out):
        super(MyModel, self).__init__(
            l1=L.Linear(None, n_out),
        )

    def __call__(self, x):
        return self.l1(x)


def main():
    epoch = 100
    batch_size = 128

    # Load mnist data
    # http://docs.chainer.org/en/latest/reference/datasets.html
    train, test = chainer.datasets.get_mnist()

    # Define iterators.
    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

    # Initialize model: Loss function defaults to softmax_cross_entropy.
    # Can keep same model used in linear regression.
    model = L.Classifier(MyModel(10))

    # Define optimizer (SGD, Adam, RMSProp, etc)
    # http://docs.chainer.org/en/latest/reference/optimizers.html
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)

    # Set up trainer
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch, 'epoch'))

    # Evaluate the model at end of each epoch
    trainer.extend(extensions.Evaluator(test_iter, model))

    # Helper functions (extensions) to monitor progress on stdout.
    report_params = [
        'epoch',
        'main/loss',
        'validation/main/loss',
        'main/accuracy',
        'validation/main/accuracy',
        'elapsed_time'
    ]
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_params))
    trainer.extend(extensions.ProgressBar())

    # Run trainer
    trainer.run()


if __name__ == "__main__":
    main()


"""
Expected output.

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
...
90          0.289549    0.282843              0.919443       0.920293                  153.543
91          0.289916    0.282415              0.919027       0.920688                  155.247
92          0.288972    0.282059              0.919655       0.920589                  156.955
93          0.288909    0.281913              0.919426       0.920688                  158.662
94          0.288564    0.281732              0.91946        0.920886                  160.377
95          0.28832     0.281716              0.919826       0.921084                  162.081
96          0.287921    0.281607              0.919538       0.920688                  163.785
97          0.287806    0.281264              0.919759       0.921381                  165.491
98          0.287321    0.281151              0.919926       0.921479                  167.2
99          0.287164    0.280869              0.919926       0.921084                  168.897
100         0.286772    0.280792              0.920072       0.921282                  170.597
"""
