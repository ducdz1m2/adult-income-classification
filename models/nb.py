from sklearn.naive_bayes import MultinomialNB, GaussianNB

class Naive_Bayes_Model:
    def __init__(self, **kwargs):
        self.model = MultinomialNB(**kwargs)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)