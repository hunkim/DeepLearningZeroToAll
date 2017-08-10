#!/usr/bin/env python
# Lab 10 MNIST and MLP with dropout

import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import training, Variable
from chainer import datasets, iterators, optimizers
from chainer import Chain
from chainer.training import extensions


class ModernMLP(chainer.Chain):
    # Define model to be called later by L.Classifer()
    # Basic MLP

    def __init__(self, n_units, n_out):
        super(ModernMLP, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out)
        )

    def __call__(self, x):
        # Add dropout, and use ReLU for activation function.
        # dropout:
        # This function drops input elements randomly with probability
        # ``ratio`` and scales the remaining elements by factor
        # ``1 / (1 - ratio)``. In testing mode, it does nothing and
        # just returns ``x``.
        # source: http://docs.chainer.org/en/latest/_modules/chainer/functions/noise/dropout.html?highlight=dropout
        h = F.dropout(F.relu(self.l1(x)), ratio=0.3, train=True)
        h = F.dropout(F.relu(self.l2(h)), ratio=0.3, train=True)
        return self.l3(h)


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
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
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
    model = L.Classifier(ModernMLP(625, 10))

    # Set up GPU usage if necessary. args.gpu is a condition as well as an
    # identification when passed to get_device().
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Define optimizer (SGD, Adam, RMSprop, etc)
    # http://docs.chainer.org/en/latest/reference/optimizers.html
    # RMSprop default parameter setting:
    # lr=0.01, alpha=0.99, eps=1e-8
    optimizer = chainer.optimizers.RMSprop()
    optimizer.setup(model)

    # Set up trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))

    # Evaluate the model at end of each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

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

    # Here we add a bit more boiler plate code to help in output of useful
    # information in related to training. Very intuitive and great for post
    # analysis.
    # source:
    # https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'],
                'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    if args.resume:
        # Resume from a snapshot (NumPy NPZ format and HDF5 format available)
        # http://docs.chainer.org/en/latest/reference/serializers.html
        chainer.serializers.load_npz(args.resume, trainer)

    # Run trainer
    trainer.run()


if __name__ == "__main__":
    main()


"""
Expected output with 1 gpu.

epoch       main/loss   validation/main/loss  main/accuracy validation/main/accuracy  elapsed_time
...
90          0.217452    0.965264              0.958189       0.941456                 294.61
91          0.196134    1.14531               0.959089       0.944917                 297.859
92          0.203648    0.956059              0.957148       0.943928                 301.109
93          0.20284     1.02199               0.960021       0.948378                 304.362
94          0.195888    1.18072               0.958905       0.945609                 307.619
95          0.199831    1.2245                0.958356       0.94195                  310.879
96          0.200486    1.10434               0.960186       0.943038                 314.151
97          0.202059    1.43919               0.960421       0.943335                 317.447
98          0.221666    0.947955              0.959305       0.946994                 320.745
99          0.200717    1.35896               0.961504       0.943137                 324.038
100         0.182234    0.935365              0.962039       0.946301                 327.323
"""
