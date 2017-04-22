#!/usr/bin/env python
# Lab 10 MNIST and NN

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions


class MLP(chainer.Chain):
    # Define model to be called later by L.Classifer()
    # Basic MLP

    def __init__(self, n_unit, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(None, n_unit),
            l2=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)


def main():
    # Introduce argparse for clarity and organization.
    # Starting to use higher capacity models, thus set up for GPU.
    parser = argparse.ArgumentParser(description='Chainer-Tutorial: MLP')
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of times to train on data set')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID: -1 indicates CPU')
    args = parser.parse_args()

    # Load mnist data
    # http://docs.chainer.org/en/latest/reference/datasets.html
    train, test = chainer.datasets.get_mnist()

    # Define iterators.
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    # Initialize model: Loss function defaults to softmax_cross_entropy.
    # 784 is dimension of the inputs, 625 is n_units in hidden layer
    # and 10 is the output dimension.
    model = L.Classifier(MLP(625, 10))

    # Set up GPU usage if necessary. args.gpu is a condition as well as an
    # identification when passed to get_device().
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Define optimizer (SGD, Adam, RMSProp, etc)
    # http://docs.chainer.org/en/latest/reference/optimizers.html
    optimizer = chainer.optimizers.SGD()
    optimizer.setup(model)

    # Set up trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # Evaluate the model at end of each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

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
# Expected output with 1 gpu.

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy  elapsed_time
...
90          0.285929    0.277819              0.918177       0.921084                  198.132
91          0.285159    0.277838              0.918077       0.920194                  200.3
92          0.28487     0.277246              0.918453       0.9196                    202.472
93          0.284443    0.276658              0.918643       0.920589                  204.645
94          0.283882    0.276925              0.918877       0.920985                  206.818
95          0.283553    0.276153              0.91906        0.920688                  209.031
96          0.283272    0.275503              0.919071       0.921282                  211.219
97          0.282494    0.274468              0.91941        0.921084                  213.428
98          0.282246    0.274534              0.91936        0.921381                  215.617
99          0.2818      0.274671              0.919543       0.921875                  217.821
100         0.281342    0.27406               0.919772       0.922567                  220.023
"""
