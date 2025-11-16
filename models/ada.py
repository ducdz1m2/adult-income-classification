# models/ada.py
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost_Model:
    def __init__(self, **kwargs):
        self.model = AdaBoostClassifier(**kwargs)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
