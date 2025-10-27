import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from preprocessing import PreProcessing

sys.stdout.reconfigure(encoding='utf-8')

class ModelTuner:
    def __init__(self, X_train, y_train, cv=5, scoring='f1'):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.scoring = scoring

    def tune_decision_tree(self):
        param_grid = {
            'max_depth': [4, 6, 8, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'criterion': ['gini', 'entropy']
        }
        model = DecisionTreeClassifier(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        return grid.best_estimator_, grid.best_params_, grid.best_score_

    def tune_knn(self):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        model = KNeighborsClassifier()
        grid = GridSearchCV(model, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        return grid.best_estimator_, grid.best_params_, grid.best_score_

    def tune_nb(self):
        param_grid = {'var_smoothing': np.logspace(-9, -1, 9)}
        model = GaussianNB()
        grid = GridSearchCV(model, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        return grid.best_estimator_, grid.best_params_, grid.best_score_

    def tune_rf(self):
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        model = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=20, cv=self.cv,
            scoring=self.scoring, n_jobs=-1, random_state=42
        )
        random_search.fit(self.X_train, self.y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


# 1️⃣ Đọc dữ liệu
df = pd.read_csv("data/adult.data", header=None, skipinitialspace=True, encoding='utf-8')

# 2️⃣ Tiền xử lý
pre = PreProcessing(df)
X, y = pre.process()
print(f">>> Dữ liệu sau tiền xử lý: X={X.shape}, y={y.shape}")

# 3️⃣ Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Gọi tuner
tuner = ModelTuner(X_train, y_train, cv=5, scoring='f1')

print("\n=== 🔎 Tuning Decision Tree ===")
best_tree, params_tree, score_tree = tuner.tune_decision_tree()
print("Best params:", params_tree)
print("CV F1:", score_tree)

print("\n=== 🔎 Tuning KNN ===")
best_knn, params_knn, score_knn = tuner.tune_knn()
print("Best params:", params_knn)
print("CV F1:", score_knn)

print("\n=== 🔎 Tuning Naive Bayes ===")
best_nb, params_nb, score_nb = tuner.tune_nb()
print("Best params:", params_nb)
print("CV F1:", score_nb)

print("\n=== 🔎 Tuning Random Forest ===")
best_rf, params_rf, score_rf = tuner.tune_rf()
print("Best params:", params_rf)
print("CV F1:", score_rf)
