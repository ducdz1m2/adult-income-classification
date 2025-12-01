from sklearn.neighbors import KNeighborsClassifier

class KNN_Model:
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
