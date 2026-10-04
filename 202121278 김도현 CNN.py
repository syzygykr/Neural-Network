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
        self.b = self.b - learning_rate * grad_h.sum((0, 1, 2), keepdims=True).reshape(1, -1) / (h * w)
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
                pooling_grad = np.stack(index, -1)  # (b, c, stride*stride)
                pooling_grad = pooling_grad.reshape(b, c, self.stride, self.stride)
                pooling_grad = pooling_grad.transpose(0, 2, 3, 1)
                grad_next[:, i*self.stride:(i+1)*self.stride,
                             j*self.stride:(j+1)*self.stride, :] = pooling_grad
        return grad_next

F = [perceptron(1, 32, 5), pooling(), perceptron(32, 64, 5), pooling(), perceptron(64, 10, 4, True)]

X_train, Y_train, X_test, Y_test = load_data()

lr = 1e-2
Loss = []
arr_cost = [[] for _ in range(10)]
batch_size = 128

N = len(X_train)

# 그래프 출력을 위한 설정(창 생성, 축 레이블, 범례 설정 및 초기화)
plt.ion()
fig_loss, ax_loss = plt.subplots(num='Training Loss')
loss_line, = ax_loss.plot([], [], label='Total loss')
ax_loss.set_xlabel('Training step')
ax_loss.set_ylabel('Loss')
ax_loss.legend(loc='upper right')

fig_cost, ax_cost = plt.subplots(num='Cost by Class')
class_lines = [
    ax_cost.plot([], [], label=f'for class {class_id}')[0]
    for class_id in range(10)
]
ax_cost.set_xlabel('Training step')
ax_cost.set_ylabel('Cost')
ax_cost.legend(loc='upper right')
plt.show(block=False)

# 학습
for epoch in range(10):
    X, Y = [], []
    idx = np.arange(N)
    np.random.shuffle(idx)

    for id_ in idx:
        X.append(X_train[id_])
        Y.append(Y_train[id_])

        if len(Y) == batch_size:
            #update parameters
            X = np.stack(X, 0)
            Y = np.stack(Y).reshape(-1)
            Label = np.zeros((batch_size, 10), dtype=np.float32)
            Label[np.arange(batch_size), Y] = 1.0
            # Forward pass
            Y_pred = X.copy()
            for p in F:
                Y_pred = p.forward(Y_pred)
            # Compute Loss (or cost)
            eps = 1e-12
            sample_cost = -np.sum(Label * np.log(Y_pred + eps), axis=1)
            Loss.append(sample_cost.mean())
            for class_id in range(10):
                class_cost = sample_cost[Y == class_id]
                # Keep the same training step for all classes.
                arr_cost[class_id].append(class_cost.mean() if class_cost.size else np.nan)

            # Update parameters
            grad = (Y_pred - Label) / batch_size
            for p in F[::-1]:
                grad = p.backward(grad, lr)
            X, Y = [], []

            # 그래프 실시간 업데이트
            steps = np.arange(len(Loss))
            loss_line.set_data(steps, Loss)
            for class_id, line in enumerate(class_lines):
                line.set_data(steps, arr_cost[class_id])
            for fig, ax in ((fig_loss, ax_loss), (fig_cost, ax_cost)):
                ax.relim()
                ax.autoscale_view()
                ax.set_ylim(bottom=0)
                fig.canvas.draw_idle()
            plt.pause(0.01)     

# 테스트셋을 통한 모델 정확도 평가
Y_pred = X_test.copy()
for p in F:
    Y_pred = p.forward(Y_pred)
Y_pred = np.argmax(Y_pred, -1)
print('ACC: {:.2f}'.format(np.mean(Y_pred == Y_test.reshape(-1)) * 100))

# 창을 종료하지 말고 대기
plt.ioff()
plt.show(block=True)
