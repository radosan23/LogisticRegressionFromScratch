import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    @staticmethod
    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        t = np.dot(row, coef_[1:]) + coef_[0] if self.fit_intercept else np.dot(row, coef_)
        return self.sigmoid(t)


def z_standard(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


def main():
    # prepare data
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X, y = z_standard(X[['worst concave points', 'worst perimeter']].values), y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

    model = CustomLogisticRegression()
    coefs = np.array([0.77001597, -2.12842434, -2.39305793])
    prob = model.predict_proba(X_test[:10], coefs)
    print(prob.tolist())


if __name__ == '__main__':
    main()
