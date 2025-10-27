from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, **kwargs):
        # Có thể truyền max_depth, criterion, random_state...
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
