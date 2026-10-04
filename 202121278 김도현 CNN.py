import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

# 데이터 로더
def load_data():
    train_data = MNIST(root='./data', train=True, download=True)
    test_data = MNIST(root='./data', train=False, download=True)

    # image: (N, 28, 28) -> (N, 784)
    X_train = train_data.data.numpy().reshape(-1, 28, 28, 1).astype(np.float32)
    X_test = test_data.data.numpy().reshape(-1, 28, 28, 1).astype(np.float32)

    # normalize [0, 255] -> [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    # labels
    Y_train = train_data.targets.numpy()
    Y_test = test_data.targets.numpy()

    # visualize the first sample and check its label
    plt.imshow(X_train[0].reshape(28, 28))
    plt.xlabel("Number: {}".format(Y_train[0]))
    plt.show()

    return X_train, Y_train, X_test, Y_test

# 퍼셉트론 정의
class perceptron(): # one convolutional layer
    def __init__(self, c_in, c_out, patch_sz, is_final=False):
        # c_in is the number of input neuron
        # c_out is the number of output neuron
        self.patch_sz = patch_sz
        self.c_out = c_out
        self.w = np.random.rand(c_in * patch_sz * patch_sz, c_out) * np.sqrt(2. / c_in)
        self.b = np.zeros([1, c_out])
        self.is_final = is_final

    def forward(self, x):
        self.x = x.copy()
        batch_sz, h, w, c = x.shape
        h_out = h - (self.patch_sz - 1)
        w_out = w - (self.patch_sz - 1)
        self.h = np.zeros([batch_sz, h_out, w_out, self.c_out])
        for i in range(h_out):
            for j in range(w_out):
                h = x[:, i:i+self.patch_sz, j:j+self.patch_sz, :]
                h = h.reshape(batch_sz, -1) @ self.w + self.b
                self.h[:, i, j, :] = h

        if self.is_final:
            # softmax function
            exp_h = self.h - self.h.max(axis=-1, keepdims=True)
            exp_h = np.exp(exp_h)
            y_pred = exp_h / (np.sum(exp_h, -1, keepdims=True) + 1e-6)
        else:
            # Relu
            y_pred = np.maximum(self.h, 0)
        return y_pred


    def backward(self, grad, learning_rate):
        # Compute gradient
        grad_h = grad.copy()
        batch_sz, h, w, c = grad_h.shape
        if not self.is_final:
            grad_h[self.h < 0] = 0
        grad_next = np.zeros_like(self.x)
        overlap = np.zeros_like(self.x)
        grad_w = np.zeros_like(self.w)
        for i in range(h):
            for j in range(w):
                x = self.x[:, i:i+self.patch_sz, j:j+self.patch_sz, :].reshape(batch_sz, -1)
                grad_w += x.T @ grad_h[:, i, j, :]
                grad_next[:, i:i+self.patch_sz, j:j+self.patch_sz, :] += \
                    (grad_h[:, i, j, :] @ self.w.T).reshape(batch_sz, self.patch_sz, self.patch_sz, -1)
                overlap[:, i:i+self.patch_sz, j:j+self.patch_sz, :] = 1

        # Update parameters
        self.w = self.w - learning_rate * grad_w / (h * w)
        self.b = self.b - learning_rate * grad_h.mean((0, 1, 2), keepdims=True)
        return grad_next / overlap

# 풀링
class pooling():
    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, x):
        b, h, w, c = x.shape
        h = h // self.stride
        w = w // self.stride

        y = np.zeros([b, h, w, c])
        self.idx = np.zeros([b, h, w, c])

        for i in range(h):
            for j in range(w):
                candidate = []
                for ii in range(self.stride):
                    for jj in range(self.stride):
                        candidate.append(x[:, 2*i+ii, 2*j+jj])

                candidate = np.stack(candidate, -1) # b x c x 4
                y[:, i, j, :] = np.max(candidate, axis=-1)
                self.idx[:, i, j, :] = np.argmax(candidate, axis=-1)
        return y

    def backward(self, grad, learning_rate):
        b, h, w, c = grad.shape
        grad_next = np.zeros([b, h * self.stride, w * self.stride, c])

        for i in range(h):
            for j in range(w):
                index = self.idx[:, i, j, :]
                index = [(index == k) * grad[:, i, j, :] for k in range(self.stride*self.stride)]
    
                # Reshape the gradient for the pooling layer
                pooling_grad = np.stack(index, -1).reshape(b, self.stride, self.stride, -1)
                grad_next[:, i*self.stride:(i+1)*self.stride,
                             j*self.stride:(j+1)*self.stride, :] = pooling_grad
        return grad_next

F = [perceptron(1, 32, 5), pooling(), perceptron(32, 64, 5), pooling(), perceptron(64, 10, 4, True)]