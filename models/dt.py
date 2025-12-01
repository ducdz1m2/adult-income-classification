from sklearn.tree import DecisionTreeClassifier

class DecisionTree_Model:
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
