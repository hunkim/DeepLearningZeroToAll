"""
Logistic Regression

y = sigmoid(X @ W + b)

"""
import numpy as np

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X_train = np.array(x_data, dtype=np.float32)
y_train = np.array(y_data).reshape(-1, 1)

N = X_train.shape[0]
D = X_train.shape[1]

C = 1
LEARNING_RATE = 0.1
MAX_ITER = 1000

W = np.random.standard_normal((D, C))
b = np.zeros((C,))


def sigmoid(x):
    """Sigmoid function """
    sigmoid = 1 / (1 + np.exp(-x))

    return sigmoid


def sigmoid_cross_entropy(logit, labels):
    """Compute a binary cross entropy loss

    z = logit = X @ W + b
    p = sigmoid(z)
    loss_i = y * - log(p) + (1 - y) * - log(1 - p)

    Args:
        logit (2-D Array): Logit array of shape (N, 1)
        labels (2-D Array): Binary Label array of shape (N, 1)

    Returns:
        float: mean(loss_i)
    """
    assert logit.shape == (N, C)
    assert labels.shape == (N, C)

    probs = sigmoid(logit)
    loss_i = labels * -np.log(probs + 1e-8)
    loss_i += (1 - labels) * -np.log(1 - probs + 1e-8)

    loss = np.mean(loss_i)

    return loss


def grad_sigmoid_cross_entropy(logit, labels):
    """Returns

    d_loss_i       d_sigmoid
    --------   *   ---------
    d_sigmoid      d_z

    z = logit = X * W + b

    Args:
        logit (2-D Array): Logit array of shape (N, 1)
        labels (2-D Array): Binary Label array of shape (N, 1)

    Returns:
        2-D Array: Backpropagated gradients of loss (N, 1)
    """
    return sigmoid(logit) - labels


def affine_forward(X, W, b):
    """Returns a logit

    logit = X @ W + b

    Args:
        X (2-D Array): Input array of shape (N, D)
        W (2-D Array): Weight array of shape (D, 1)
        b (1-D Array): Bias array of shape (1,)

    Returns:
        logit (2-D Array): Logit array of shape (N, 1)
    """
    return np.dot(X, W) + b


for i in range(MAX_ITER):

    logit = affine_forward(X_train, W, b)
    loss = sigmoid_cross_entropy(logit, y_train)
    d_loss = grad_sigmoid_cross_entropy(logit, y_train)

    d_W = np.dot(X_train.T, d_loss) / N
    d_b = np.sum(d_loss) / N

    W -= LEARNING_RATE * d_W
    b -= LEARNING_RATE * d_b

    if i % (MAX_ITER // 10) == 0:
        prob = affine_forward(X_train, W, b)
        prob = sigmoid(prob)
        pred = prob > 0.5
        acc = (pred == y_train).mean()

        print("[Step: {:5}] Loss: {:<5.3} Accuracy: {:>5.2%}".format(i, loss, acc))

"""
[Step:     0] Loss: 2.35  Accuracy: 50.00%
[Step:   100] Loss: 0.523 Accuracy: 83.33%
[Step:   200] Loss: 0.435 Accuracy: 83.33%
[Step:   300] Loss: 0.368 Accuracy: 83.33%
[Step:   400] Loss: 0.316 Accuracy: 83.33%
[Step:   500] Loss: 0.275 Accuracy: 83.33%
[Step:   600] Loss: 0.243 Accuracy: 100.00%
[Step:   700] Loss: 0.217 Accuracy: 100.00%
[Step:   800] Loss: 0.196 Accuracy: 100.00%
[Step:   900] Loss: 0.178 Accuracy: 100.00%
"""
