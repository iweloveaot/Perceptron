from perceptron import SingleLayerPerceptron
from utils import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np


# 3.1
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2, random_state=42,
                           n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('=== Обучение однослойного перцептрона ===')
perc = SingleLayerPerceptron(2)
perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32)
y_pred = perc.predict(X_test)

# 3.2
loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"])

# 3.3
train_acc = accuracy_score(y_train, perc.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f'Точность на обучающей выборке: {train_acc}\nТочность на тестовой выборке: {test_acc}')

# 3.4
dividing_line_graph(X, y, perc)

# 4
n_experiment(X_train, X_test, y_train, y_test)
batch_experiment(X_train, X_test, y_train, y_test)
w_experiment(X_train, X_test, y_train, y_test)


# Дополнительные задания
# 1
print('=== Обучение на сгенерированных синтетических данных ===')
data_type = [('linear', '=== Линейно разделимые данные ==='), ('circle', '=== Окружность ==='), ('xor', '=== XOR ===')]
for elem in data_type:
    print(elem[1])
    X, y = generate_synthetic_data(elem[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    perc = SingleLayerPerceptron(2)
    perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32)
    y_pred = perc.predict(X_test)

    loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"], "linear_data.png")

    train_acc = accuracy_score(y_train, perc.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    print(f'Точность на обучающей выборке: {train_acc}\nТочность на тестовой выборке: {test_acc}')
    dividing_line_graph(X, y, perc, f'dividing_{elem[0]}')

# 2
print('=== Сравнение бинарной кросс-энтропии и hinge loss ===')
print("Обучение с бинарной кросс-энтропией")
perc = SingleLayerPerceptron(2, loss="bce")
perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32)
y_pred = perc.predict(X_test)
print(y_test)
print(y_pred)
loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"])
train_acc = accuracy_score(y_train, perc.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f'Точность на обучающей выборке: {train_acc}\nТочность на тестовой выборке: {test_acc}\nЛосс: {perc.loss_history["train_loss"][-1]} {perc.loss_history["val_loss"][-1]}')
dividing_line_graph(X, y, perc)

print("Обучение с hinge loss")
perc = SingleLayerPerceptron(2, loss="hinge")
perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32)
y_pred = perc.predict(X_test)
print(y_test)
print(y_pred)
loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"])
train_acc = accuracy_score(y_train, perc.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f'Точность на обучающей выборке: {train_acc}\nТочность на тестовой выборке: {test_acc}\nЛосс: {perc.loss_history["train_loss"][-1]} {perc.loss_history["val_loss"][-1]}')
dividing_line_graph(X, y, perc)

l2s = [0.0, 0.001, 0.1, 2, 10]
for l2 in l2s:
    print(f"Исследуем обучение с l2-регулязацией: l = {l2}")
    perc = SingleLayerPerceptron(2, l2=l2)
    perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32)
    y_pred = perc.predict(X_test)
    print(y_test)
    print(y_pred)
    loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"])
    train_acc = accuracy_score(y_train, perc.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    print(
        f'Точность на обучающей выборке: {train_acc}\nТочность на тестовой выборке: {test_acc}\nЛосс: {perc.loss_history["train_loss"][-1]} {perc.loss_history["val_loss"][-1]}')
    dividing_line_graph(X, y, perc)
    print()

# 3
print('=== Исследуем метрики ===')
perc = SingleLayerPerceptron(2)
perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32)
y_pred = perc.predict(X_test)
y_prob = perc.forward(X_test)
print(f'Precision score: {precision_score(y_test, y_pred):.4f}')
print(f'Recall score: {recall_score(y_test, y_pred):.4f}')
print(f'F1-score: {f1_score(y_test, y_pred):.4f}')
print(f'ROC-AUC score: {roc_auc_score(y_test, y_prob):.4f}')

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(13, 13))
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_score(y_test, y_prob):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid()
plt.legend()
plt.show()
plt.close()

errors = y_test != y_pred
plt.figure(figsize=(13, 13))
plt.scatter(X_test[~errors, 0], X_test[~errors, 1], c=y_test[~errors], cmap='coolwarm', alpha=0.6, label='классифицированны верно')
plt.scatter(X_test[errors, 0], X_test[errors, 1], c='black', marker='x', s=80, label='классифицированны ошибочно')
plt.title('Ошибочно классифоцированные точки')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.show()
plt.close()

# 4
print('=== Исследуем SGD Momentum ===')
betas = [0.0, 0.5, 0.9, 0.99]
for beta in betas:
    perc = SingleLayerPerceptron(2)
    perc.fit(X_train, y_train, X_test, y_test, 100, 0.1, 32, beta=beta)
    y_pred = perc.predict(X_test)

    loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"])

    train_acc = accuracy_score(y_train, perc.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    print(f'Точность на обучающей выборке: {train_acc}\nТочность на тестовой выборке: {test_acc}\nЛосс: {perc.compute_loss(X_test, y_test)}')

# 5
print("=== 5-кратная кросс-валидация ===")
speeds = [0.001, 0.01, 0.1, 0.5]
batch_sizes = [16, 32, 64]
kf = KFold(n_splits=5, shuffle=True)
best_score, best_params = -1, None

for n in speeds:
    for bs in batch_sizes:
        scores = []
        for tr_idx, val_idx in kf.split(X):
            X_tr_cv, X_val_cv = X[tr_idx], X[val_idx]
            y_tr_cv, y_val_cv = y[tr_idx], y[val_idx]
            sc = StandardScaler()
            X_tr_sc, X_val_sc = sc.fit_transform(X_tr_cv), sc.transform(X_val_cv)
            p = SingleLayerPerceptron(2)
            p.fit(X_tr_sc, y_tr_cv, X_val_sc, y_val_cv, 100, n, bs)
            scores.append(accuracy_score(y_val_cv, p.predict(X_val_sc)))
        mean_s, std_s = np.mean(scores), np.std(scores)
        print(f"lr={n:.3f} | batch={bs:3d} -> Acc: {mean_s:.4f} ± {std_s:.4f}")
        if mean_s > best_score:
            best_score, best_params = mean_s, (n, bs)

print(f"\nЛучшие гиперпараметры: lr={best_params[0]}, batch_size={best_params[1]} (Mean Acc: {best_score:.4f})")

print('Финальное обучение с лучшими гиперпараметрами')
perc = SingleLayerPerceptron(2)
perc.fit(X_train, y_train, X_test, y_test, 100, best_params[0], best_params[1])
y_pred = perc.predict(X_test)

loss_graph(perc.loss_history["train_loss"], perc.loss_history["val_loss"])

train_acc = accuracy_score(y_train, perc.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
print(f'Точность на обучающей выборке: {train_acc}\nТочность на тестовой выборке: {test_acc}')

dividing_line_graph(X, y, perc)




TODO: "!!!!!!!!!!!!!!! добавить random_seed !!!!!!!!!!!!!!!"

