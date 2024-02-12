import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = None

    @staticmethod
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X, y):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.coef_ = np.zeros(X.shape[1])
        for _ in range(self.n_epoch):
            for i, row in enumerate(X):
                pred = self.predict_proba(row, self.coef_)
                grad = (pred - y[i]) * pred * (1 - pred) * row
                self.coef_ -= self.l_rate * grad

    def fit_log_loss(self, X, y):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.coef_ = np.zeros(X.shape[1])
        for _ in range(self.n_epoch):
            for i, row in enumerate(X):
                pred = self.predict_proba(row, self.coef_)
                grad = (pred - y[i]) / X.shape[0] * row
                self.coef_ -= self.l_rate * grad

    def predict(self, X, cut_off=0.5):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        prediction = self.predict_proba(X, self.coef_)
        prediction = (prediction >= cut_off).astype(int)
        return prediction


def z_standard(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def main():
    # prepare data
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X, y = z_standard(X[['worst concave points', 'worst perimeter', 'worst radius']].values), y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

    model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    model.fit_log_loss(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    info = {'coef_': model.coef_.tolist(), 'accuracy': accuracy}
    print(info)


if __name__ == '__main__':
    main()
