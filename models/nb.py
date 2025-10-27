from sklearn.naive_bayes import GaussianNB

class Naive_Bayes_Model:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_model(self):
        return self.model