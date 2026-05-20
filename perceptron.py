import numpy as np


class SingleLayerPerceptron:
    def __init__(self, n_features, w_init_type="small", loss="bce", l2=0.0):
        np.random.seed(42)
        self.n_features = n_features
        self.w = []
        if w_init_type == "zeros":
            self.w = np.zeros(n_features)
        if w_init_type == "small":
            self.w = np.random.sample(n_features)
        if w_init_type == "large":
            self.w = np.random.sample(n_features) * 10.0
        self.b = 0.0
        self.loss = loss
        self.l2 = l2
        self.loss_history = {'train_loss': [], 'val_loss': []}

        self.v_w = np.zeros(n_features)
        self.v_b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def compute_loss(self, X, y):
        Z = np.dot(X, self.w) + self.b
        if self.loss == "bce":
            y_pred = self.sigmoid(Z)
            loss = -np.mean((y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))
        elif self.loss == "hinge":
            y_true_new = np.where(y <= 0, -1, 1)
            loss = np.mean(np.maximum(0, 1 - y_true_new * Z))
        else:
            raise ValueError("Доступные типы: 'bce', 'hinge'")

        l2_reg = self.l2 * np.sum(self.w ** 2) / 2
        return loss + l2_reg

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size, beta=0.0):
        np.random.seed(42)
        for epoch in range(epochs):
            n_samples = X_train.shape[0]

            indexes = np.random.permutation(n_samples)
            X_train = X_train[indexes]
            y_train = y_train[indexes]

            for i in range(0, n_samples, batch_size):
                X_train_batch = X_train[i:i+batch_size]
                y_train_batch = y_train[i:i+batch_size]

                z = np.dot(X_train_batch, self.w) + self.b
                if self.loss == "bce":
                    y_pred = self.sigmoid(z)
                    dz = y_pred - y_train_batch
                elif self.loss == "hinge":
                    y_true = np.where(y_train_batch <= 0, -1, 1)
                    mask = (y_true * z) < 1
                    dz = -y_true * mask
                else:
                    raise ValueError("undefined loss type")

                dw = (np.dot(X_train_batch.T, dz) / n_samples) + self.l2 * self.w
                db = np.mean(dz)

                self.v_w = beta * self.v_w + lr * dw
                self.v_b = beta * self.v_b + lr * db
                self.w -= self.v_w
                self.b -= self.v_b

            self.loss_history['train_loss'].append(self.compute_loss(X_train, y_train))
            self.loss_history['val_loss'].append(self.compute_loss(X_val, y_val))

    def predict(self, X):
        if self.loss == 'hinge':
            z = np.dot(X, self.w) + self.b
            return np.where(z > 0, 1, 0)
        y_pred = self.forward(X)
        res = np.array(y_pred >= 0.5, int)
        return res
