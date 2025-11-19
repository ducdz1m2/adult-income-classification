import pandas as pd

from data_plot import DataVisualizer
from evaluate import Evaluator
from models.ada import AdaBoost_Model
from models.dt import DecisionTree_Model
from models.knn import KNN_Model
from models.nb import NaiveBayes_Model
from models.rf import RandomForest_Model
from models.svm import SVM_Model
from plot import Plotter
from preprocessing import PreProcessing

# ---- Đọc dữ liệu ----
train_file = "data/adult.data"
test_file = "data/adult.test"

pre_train = PreProcessing(train_file)
X_train, y_train = pre_train.run()

pre_test = PreProcessing(test_file, is_test=True)
# dùng cùng encoder với train
pre_test.encoders = pre_train.encoders
X_test, y_test = pre_test.run()

print(f">>> Train: X={X_train.shape}, y={y_train.shape}")
print(f">>> Test: X={X_test.shape}, y={y_test.shape}")

# ---- Vẽ biểu đồ trực quan ----
visualizer = DataVisualizer(pd.read_csv(train_file, header=None))
visualizer.plot_numeric_pair(["age", "hours-per-week"])
visualizer.plot_categorical_pair(["workclass", "education"], top_n=6)
visualizer.plot_income_vs_feature("occupation")
visualizer.plot_box_numeric_vs_income("age")
visualizer.plot_pie_income()

# ---- Huấn luyện mô hình ----

models = {
    "Decision Tree": DecisionTree_Model(
        criterion="entropy", max_depth=12, min_samples_split=4, min_samples_leaf=12
    ),
    "KNN": KNN_Model(n_neighbors=11, p=2, weights="uniform"),
    "Naive Bayes": NaiveBayes_Model(),
    "Random Forest": RandomForest_Model(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    ),
    "SVM": SVM_Model(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    "AdaBoost": AdaBoost_Model(n_estimators=200, learning_rate=0.5, random_state=42),
}

for name, model in models.items():
    model.train(X_train, y_train)
    evaluator = Evaluator(model.model, X_test, y_test)
    results = evaluator.evaluate()
    print(f"\n>>> Kết quả đánh giá {name}:")
    for metric, value in results.items():
        print(f"- {metric}: {value:.4f}")

# ---- Feature importance Decision Tree ----
tree_model = models["Decision Tree"].model
feature_names = list(X_train.columns)
importances = tree_model.feature_importances_
top_n = 15
sorted_idx = importances.argsort()[-top_n:]
print("\nTop feature names & importances:")
for i in sorted_idx:
    print(f"- {feature_names[i]}: {importances[i]:.4f}")

plotter = Plotter(tree_model, X_test, y_test)
plotter.feature_importance(feature_names, top_n=top_n)
