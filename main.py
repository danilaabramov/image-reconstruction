import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn import linear_model


def read_csv(filename: str):
    with open(filename, "r") as input_file:
        arr = [line.split(",") for line in input_file if len(line) > 0]
    return arr


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, w, n):
    y_pred = np.array([sigmoid(x.reshape(1, n).dot(w)) for x in X])
    return y_pred.flatten()


def function(img, number_basis):
    M = img.load()
    size0, size1 = img.size
    m = size0 * size1 * (size0 * size1 - 1) // 2 // size0
    couples = size0 * size1 * (size0 * size1 - 1) // 2
    if number_basis == 1:
        K = [[j, i - j] if i % 2 == 0 else [i - j, j] for i in range(size0 * size1) for j in range(i + 1)]
        basis = [[(np.cos((i % size0) / (size0 - 1) * K[l][0]) * np.pi) * np.cos((i // size0) / (size1 - 1) * K[l][1]
                                                                                 * np.pi)
                  for i in range(size0 * size1)] for l in range(size0 * size1)]

    if number_basis == 2:
        basis = read_csv('basisYale32.csv')
        basis = np.array(list(map(lambda x: np.array(list(map(lambda y: int(y), basis[x[0]]))),
                                  enumerate(basis))))

    n = int(len(basis[0]) * 1)
    pca = PCA(n)
    x_test = pca.fit_transform(np.transpose(basis))
    print('Доля информации: ', sum(pca.explained_variance_ratio_))

    colors = np.array([M[i % size0, i // size0] for i in range(size0 * size1)])

    x_train, y_train = [], []
    while len(x_train) < m:
        i, j = np.random.choice(size1 * size0, 2, replace=False)
        x_train.append(x_test[i] - x_test[j])
        y_train.append(int(colors[i] > colors[j]))
    x_train, y_train = np.array(x_train), np.array(y_train)

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    prediction = predict(x_test, reg.coef_, n)

    min_img_color = min(colors)
    max_img_color = max(colors)
    max_prediction_value = max(prediction)
    min_prediction_value = min(prediction)
    prediction = [(pix - min_prediction_value) * (max_img_color - min_img_color)
                  / (max_prediction_value - min_prediction_value) + min_img_color for pix in prediction]
    faithful_couples = sum((colors[i] - colors[j]) * (prediction[i] - prediction[j]) >= 0
                           for i in range(len(prediction) - 1) for j in range(i + 1, len(prediction)))
    print("Точность предсказаний: ", faithful_couples / couples * 100, "%")

    prediction = np.array(prediction, dtype=int).reshape((size1, size0))
    prediction = [[[pix, pix, pix] for pix in row] for row in prediction.tolist()]
    plt.imshow(prediction, interpolation='none')
    plt.show()


img = Image.open("yale32.pgm")
number_basis = 2  # 1 - Фурье-базис, 2 - базис по БД лиц
function(img, number_basis)