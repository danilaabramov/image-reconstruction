import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("dev.png")
M = img.load()

size0, size1 = img.size
m = size0 * size1 * (size0 * size1 - 1) // 2
n = 1000


K = []
a = 0
b = 0
for i in range(n):
    if a == b:
        K.append([a, b])
        a += 1
        b = 0
    else:
        K.append([b, a])
        K.append([a, b])
        b += 1

X_train = []
X = []

y_train = np.array([])

for i in range(size0 * size1 - 1):
    j = i + 1
    while j < size0 * size1:
        u = np.ones((n, 1))
        u2 = np.ones((n, 1))
        for l in range(n):
            u[l] = (np.cos((i % size0) / (size0 - 1) * K[l][0]) * np.pi) * np.cos((i // size0) / (size1 - 1) * K[l][1] * np.pi)
            u2[l] = (np.cos((j % size0) / (size0 - 1) * K[l][0]) * np.pi) * np.cos((j // size0) / (size1 - 1) * K[l][1] * np.pi)
        X_train.append(u - u2)
        if M[i % size0, i // size0][0] > M[j % size0, j // size0][0]:
            y_train = np.append(y_train, 1)
        else:
            y_train = np.append(y_train, 0)
        j += 1

for i in range(size0 * size1):
    j = i + 1
    u = np.ones((n, 1))
    for l in range(n):
        u[l] = (np.cos((i % size0) / (size0 - 1) * K[l][0]) * np.pi) * np.cos((i // size0) / (size1 - 1)  * K[l][1] * np.pi)
    X.append(u)

X = np.array(X)
X_train = np.array(X_train)

def log_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0) / len(y_true)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:

    def __init__(self):
        self.losses_train = []
        self.w = np.random.randn(n, 1) * 0.001

    def train_vec(self, X, y, learningRate=0.005):
        T = 0
        while log_loss(y, self.predict(X)) > 0.5:
            Z = X.reshape(m, n).dot(self.w)
            A = sigmoid(Z)

            dw = np.sum(X.reshape(m, n) * (A.reshape(m, 1) - y.reshape(m, 1)), axis=0) / len(X)

            self.w = self.w - learningRate * dw.reshape(n, 1)
            T += 1
        print(log_loss(y, self.predict(X)), T)
    def predict(self, X):
        return np.array([sigmoid(x.reshape(1, n).dot(self.w))[0][0] for x in X])

logreg = LogisticRegression()
logreg.train_vec(X_train, y_train)

prediction = np.array(logreg.predict(X))

max = prediction[0]
for i in range(size0 * size1):
    if max < prediction[i]:
        max = prediction[i]

min = prediction[0]
for i in range(size0 * size1):
    if min > prediction[i]:
        min = prediction[i]

r = 255 / (max - min)

for i in range(size0 * size1):
    prediction[i] = (prediction[i] - min) * r

tp = np.zeros((size1, size0, 3), np.uint8)
for i in range(size1):
    for j in range(size0):
        tp[i][j] = (prediction[i * size0 + j], prediction[i * size0 + j], prediction[i * size0 + j])

plt.imshow(tp, interpolation='none')
plt.show()
