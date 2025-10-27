from sklearn.ensemble import RandomForestClassifier

class RandomForest_Model:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
    