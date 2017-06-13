import numpy as np

data = np.loadtxt("../data-04-zoo.csv",
                  delimiter=",",
                  dtype=np.float32)

X_train = data[:, :-1]
y_train = data[:, -1].astype(np.int8)
assert X_train.shape == (101, 16)
assert y_train.shape == (101,)

N, D = X_train.shape
C = np.max(y_train) + 1

y_train_onehot = np.zeros(shape=(N, C))
y_train_onehot[np.arange(N), y_train] = 1

assert C == 7, "There are 7 classes to predict"

W = np.random.standard_normal((D, C))
b = np.zeros((C,))


def affine_forward(X, W, b):
    """Returns a logit

    logit = X @ W + b

    Args:
        X (2-D Array): Input array of shape (N, D)
        W (2-D Array): Weight array of shape (D, C)
        b (1-D Array): Bias array of shape (C,)

    Returns:
        logit (2-D Array): Logit array of shape (N, C)
    """
    return np.dot(X, W) + b


def softmax(z):
    """Softmax Function

    Subtract max for numerical stability

    Args:
        z (2-D Array): Array of shape (N, C)

    Returns:
        2-D Array: Softmax output of (N, C)
    """
    z -= np.max(z)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1).reshape(-1, 1) + 1e-7

    return numerator / denominator


def softmax_cross_entropy_loss(logit, labels):
    """Returns a softmax cross entropy loss

    loss_i = - log(P(y_i | x_i))

    Args:
        logit (2-D Array): Logit array of shape (N, C)
        labels (2-D Array): Label Onehot array of shape (N, C)

    Returns:
        float: mean(loss_i)
    """
    p = softmax(logit)
    loss_i = - labels * np.log(p + 1e-8)
    return np.mean(loss_i)


def grad_softmax_cross_entropy_loss(logit, labels):
    """Returns

    d_loss_i       d_softmax
    --------   *   ---------
    d_softmax      d_z

    z = logit = X * W + b

    Args:
        logit (2-D Array): Logit array of shape (N, C)
        labels (2-D Array): Onehot label array of shape (N, C)

    Returns:
        2-D Array: Backpropagated gradients of loss (N, C)

    Notes:
        [1] Neural Net Backprop in one slide! by Sung Kim
        https://docs.google.com/presentation/d/1_ZmtfEjLmhbuM_PqbDYMXXLAqeWN0HwuhcSKnUQZ6MM/edit#slide=id.g1ec1d04b5a_1_83

    """
    return softmax(logit) - labels


def get_accuracy(logit, labels):
    """Returna an accracy

    Args:
        logit (2-D Array): Logit array of shape (N, C)
        labels (2-D Array): Onehot label array of shape (N, C)

    Returns:
        float: Accuracy
    """

    probs = softmax(logit)
    pred = np.argmax(probs, axis=1)
    true = np.argmax(labels, axis=1)

    return np.mean(pred == true)


LEARNING_RATE = 0.1
MAX_ITER = 2000
PRINT_N = 10

for i in range(MAX_ITER):

    logit = affine_forward(X_train, W, b)
    loss = softmax_cross_entropy_loss(logit, y_train_onehot)
    d_loss = grad_softmax_cross_entropy_loss(logit, y_train_onehot)

    d_W = np.dot(X_train.T, d_loss) / N
    d_b = np.sum(d_loss) / N

    W -= LEARNING_RATE * d_W
    b -= LEARNING_RATE * d_b

    if i % (MAX_ITER // PRINT_N) == 0:
        acc = get_accuracy(logit, y_train_onehot)
        print("[Step: {:5}] Loss: {:<10.5} Acc: {:.2%}".format(i, loss, acc))

"""
[Step:     0] Loss: 0.76726    Acc: 31.68%
[Step:   200] Loss: 0.057501   Acc: 87.13%
[Step:   400] Loss: 0.034893   Acc: 92.08%
[Step:   600] Loss: 0.025472   Acc: 97.03%
[Step:   800] Loss: 0.020099   Acc: 97.03%
[Step:  1000] Loss: 0.016562   Acc: 99.01%
[Step:  1200] Loss: 0.014058   Acc: 100.00%
[Step:  1400] Loss: 0.012204   Acc: 100.00%
[Step:  1600] Loss: 0.010784   Acc: 100.00%
[Step:  1800] Loss: 0.0096631  Acc: 100.00%
"""
