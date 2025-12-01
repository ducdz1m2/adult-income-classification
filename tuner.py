import pandas as pd
from sklearn.model_selection import GridSearchCV

from models.ada import AdaBoost_Model
from models.dt import DecisionTree_Model
from models.knn import KNN_Model
from models.nb import NaiveBayes_Model
from models.rf import RandomForest_Model
from models.svm import SVM_Model
from preprocessing import PreProcessing


def load_adult_test(file_path):
    df = pd.read_csv(
        file_path, header=None, skipinitialspace=True, encoding="utf-8", comment="|"
    )
    df[14] = df[14].str.strip().str.replace(".", "", regex=False)
    return df


def tune(model, param_grid, X_train, y_train):
    gs = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("\nBest params:", gs.best_params_)
    print("Best score :", gs.best_score_)
    return gs.best_params_


df_train = pd.read_csv("data/adult.data", header=None, skipinitialspace=True)
df_test = load_adult_test("data/adult.test")

pre_train = PreProcessing(df=df_train)
X_train, y_train = pre_train.run()

pre_test = PreProcessing(df=df_test, is_test=False)
pre_test.encoders = pre_train.encoders
X_test, y_test = pre_test.run()

print("Dữ liệu:", X_train.shape, y_train.shape)

models_and_params = [
    ("Decision Tree", DecisionTree_Model().model, {
        "criterion": ["gini", "entropy"],
        "max_depth": [6, 10, 12, 14, None],
        "min_samples_split": [2, 4, 10],
        "min_samples_leaf": [1, 2, 4, 8, 16],
    }),
    ("KNN", KNN_Model().model, {
        "n_neighbors": [5, 11, 17],
        "p": [1, 2],
        "weights": ["uniform", "distance"],
    }),
    ("Naive Bayes", NaiveBayes_Model().model, {}),
    ("Random Forest", RandomForest_Model().model, {
        "n_estimators": [50, 150, 250],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 10],
        "max_features": ["sqrt", "log2"],
    }),
]

for name, model, param in models_and_params:
    print(f"{name}")
    tune(model, param, X_train, y_train)