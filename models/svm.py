# models/svm.py
from sklearn.svm import SVC


class SVM_Model:
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
