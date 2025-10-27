from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, **kwargs):
        # Có thể truyền n_neighbors, weights, metric, p, v.v...
        self.model = KNeighborsClassifier(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
