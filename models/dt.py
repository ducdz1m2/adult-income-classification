from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DecisionTree:
    def __init__(self, **kwargs):
        # kwargs cho phép truyền tham số như max_depth, criterion, v.v.
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc
