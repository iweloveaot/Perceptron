import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from perceptron import SingleLayerPerceptron


def loss_graph(train_loss, val_loss, filename="new_graph.png"):
    plt.figure(figsize=(13, 13))
    plt.title("Изменение функции потерь в зависимости от эпох")
    plt.xlabel("эпоха")
    plt.ylabel("loss")
    plt.grid()
    epochs = np.arange(1, 101)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.legend(["обучающая выборка", "тестовая выборка"])
    plt.savefig(filename)
    plt.show()
    plt.close()


def dividing_line_graph(X, y, model, filename="dividing_line.png", n_samples=500):
    plt.figure(figsize=(13, 13))
    plt.title("Разделяющая граница w*X+b на фоне данных")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.grid()
    first_inds = np.where(y == 0)
    second_inds = np.where(y == 1)
    plt.scatter(X[:, 0][first_inds], X[:, 1][first_inds], c="blue")
    plt.scatter(X[:, 0][second_inds], X[:, 1][second_inds], c="red")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.legend(["0-answers", "1-answers"])
    plt.savefig(filename)
    plt.show()
    plt.close()


def n_experiment(X_train, X_test, y_train, y_test):
    print("=== Эксперимент с learning rate (0.001, 0.01, 0.5, 1) ===")
    print("Точность на обучающей выборке  | Точность на тестовой выборке  | Лосс на тестовой выборке")
    n = [0.001, 0.01, 0.5, 1]
    count = 1
    for speed in n:
        perc = SingleLayerPerceptron(2)
        perc.fit(X_train, y_train, X_test, y_test, 100, speed, 32)
        y_pred = perc.predict(X_test)

        loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"], filename=f"n_experiment_{count}.png")

        train_acc = accuracy_score(y_train, perc.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        print(f'{train_acc:.5f}                        | {test_acc:.5f}                       | {perc.loss_history["val_loss"][-1]:.5f}')
        count += 1


def batch_experiment(X_train, X_test, y_train, y_test):
    print("=== Эксперимент с размером батча (1, 16, 64, 256) ===")
    print("Точность на обучающей выборке  | Точность на тестовой выборке  | Лосс на тестовой выборке")
    sizes = [1, 16, 64, 256]
    count = 1
    for size in sizes:
        perc = SingleLayerPerceptron(2)
        perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, size)
        y_pred = perc.predict(X_test)

        loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"], filename=f"batch_size_experiment_{count}.png")

        train_acc = accuracy_score(y_train, perc.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        print(f'{train_acc:.5f}                        | {test_acc:.5f}                       | {perc.loss_history["val_loss"][-1]:.5f}')
        count += 1


def w_experiment(X_train, X_test, y_train, y_test):
    print("=== Эксперимент с инициализацией весов (нули, маленькие, от 0 до 10) ===")
    print("Точность на обучающей выборке  | Точность на тестовой выборке  | Лосс на тестовой выборке")
    inits = ["zeros", "small", "large"]
    count = 1
    for init in inits:
        perc = SingleLayerPerceptron(2, init)
        perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32)
        y_pred = perc.predict(X_test)

        loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"],
                   filename=f"weight_init_experiment_{count}.png")

        train_acc = accuracy_score(y_train, perc.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        print(f'{train_acc:.5f}                        | {test_acc:.5f}                       | {perc.loss_history["val_loss"][-1]:.5f}')
        print(perc.w)
        count += 1


def generate_synthetic_data(data_type="linear", centers=([-2, -2], [2, 2]), cov_matrix=([1.5, 0], [0, 1.5]), noise=0.0, n_samples=500, random_state=42):
    np.random.seed(random_state)
    if data_type == "linear":
        c1, c2 = np.array(centers[0]), np.array(centers[1])
        cov = np.array(cov_matrix)
        X = np.vstack([np.random.multivariate_normal(c1, cov, n_samples // 2),
                       np.random.multivariate_normal(c2, cov, n_samples - n_samples // 2)])
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples - n_samples // 2)])
    elif data_type == "circle":
        r = 4.0
        X_in = np.random.uniform(-r, r, (n_samples // 2, 2))
        mask_in = np.linalg.norm(X_in, axis=1) < r
        X_out = np.random.uniform(-r * 1.5, r * 1.5, (n_samples - n_samples // 2, 2))
        mask_out = np.linalg.norm(X_out, axis=1) > r
        X = np.vstack([X_in[mask_in], X_out[mask_out]])
        y = np.hstack([np.zeros(len(X_in[mask_in])), np.ones(len(X_out[mask_out]))])
        # Доводим до нужного размера
        if len(X) < n_samples:
            extra = np.random.randn(n_samples - len(X), 2)
            X = np.vstack([X, extra])
            y = np.hstack([y, np.random.randint(0, 2, n_samples - len(y))])
    elif data_type == "xor":
        X = np.random.uniform(-3, 3, (n_samples, 2))
        y = np.array([0 if (x[0] > 0) == (x[1] > 0) else 1 for x in X])
    else:
        raise ValueError("Доступные типы: 'linear', 'circle', 'xor'")

    if noise > 0:
        flip = np.random.random(len(y)) < noise
        y[flip] = 1 - y[flip]
    return X, y