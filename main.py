import pandas as pd
from sklearn.model_selection import train_test_split

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

train_file = "data/adult.data"
test_file = "data/adult.test"

models = {
    "Decision Tree": DecisionTree_Model(
        criterion="entropy", max_depth=10, min_samples_split=10, min_samples_leaf=1
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
    ),
}

all_runs = []

results_store = {name: [] for name in models.keys()}

for i in range(1, 11):
    print("=============================\nLan chay thu " + str(i))

    pre_train = PreProcessing(train_file)
    X_train, y_train = pre_train.run()

    pre_test = PreProcessing(test_file, is_test=True)
    pre_test.encoders = pre_train.encoders
    X_test, y_test = pre_test.run()

    X_final = pd.concat([X_train, X_test],  ignore_index=True)
    y_final = pd.concat([y_train, y_test],  ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42 + i
    )

    print(f">>> Train: X={X_train.shape}, y={y_train.shape}")
    print(f">>> Test: X={X_test.shape}, y={y_test.shape}")
    for name, model in models.items():

        model.train(X_train, y_train)

        evaluator = Evaluator(model.model, X_test, y_test)
        print("\nKet qua mo hinh: " + name)
        results = evaluator.evaluate()

        acc = results.get("accuracy") if results is not None else None
        prec = results.get("precision") if results is not None else None
        rec = results.get("recall") if results is not None else None
        f1 = None
        if results is not None:
            if "f1_score" in results:
                f1 = results.get("f1_score")
            else:
                f1 = results.get("f1")

        run_row = {
            "run": i,
            "model": name,
            "accuracy": float(acc) if acc is not None else None,
            "precision": float(prec) if prec is not None else None,
            "recall": float(rec) if rec is not None else None,
            "f1": float(f1) if f1 is not None else None,
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
        }
        all_runs.append(run_row)
        results_store[name].append(run_row)

all_runs_df = pd.DataFrame(all_runs)
all_runs_df.to_csv("all_runs.csv", index=False)
print("Saved detailed runs to all_runs.csv")

summary_rows = []
for name, runs in results_store.items():
    df = pd.DataFrame(runs)
    valid = df.dropna(subset=["accuracy", "f1"])
    if valid.empty:
        summary_rows.append({
            "model": name,
            "n_valid_runs": 0,
            "avg_accuracy": None,
            "avg_precision": None,
            "avg_recall": None,
            "avg_f1": None,
        })
        continue

    avg_accuracy = valid["accuracy"].mean()
    avg_precision = valid["precision"].mean()
    avg_recall = valid["recall"].mean()
    avg_f1 = valid["f1"].mean()

    best_idx = valid["f1"].idxmax()
    best_run = valid.loc[best_idx].to_dict()

    summary_rows.append({
        "model": name,
        "n_valid_runs": int(valid.shape[0]),
        "avg_accuracy": float(avg_accuracy),
        "avg_precision": float(avg_precision),
        "avg_recall": float(avg_recall),
        "avg_f1": float(avg_f1),
        "best_run_index": int(best_run["run"]),
        "best_run_accuracy": float(best_run["accuracy"]),
        "best_run_precision": float(best_run["precision"]),
        "best_run_recall": float(best_run["recall"]),
        "best_run_f1": float(best_run["f1"]),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("summary_best.csv", index=False)
print("Saved summary (avg + best run info) to summary_best.csv")

valid_summary = summary_df.dropna(subset=["avg_f1"])
if not valid_summary.empty:
    best_overall = valid_summary.loc[valid_summary["avg_f1"].idxmax()]
    print("Model tot nhat theo avg F1:", best_overall["model"], "avg_f1=", best_overall["avg_f1"])
else:
    print("Khong co model hop le de chon best overall.")
