import pandas as pd
from data_plot import DataVisualizer
from preprocessing import PreProcessing
from models.dt import DecisionTree
from models.knn import KNN 
from models.nb import Naive_Bayes_Model
from models.rf import RandomForest_Model
from evaluate import Evaluator
from plot import Plotter
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_adult_test(file_path):
    df = pd.read_csv(file_path, header=None, skipinitialspace=True, encoding='utf-8', comment='|')
    df[14] = df[14].str.strip().str.replace('.', '', regex=False)
    return df


if __name__ == "__main__":
    # I. Đọc dữ liệu
    df_train = pd.read_csv("data/adult.data", header=None, skipinitialspace=True, encoding='utf-8')
    df_test = load_adult_test("data/adult.test")

    visualizer = DataVisualizer(df_train)
    visualizer.plot_numeric_pair()
    visualizer.plot_categorical_pair(top_n=6)
    visualizer.plot_income_vs_feature('workclass')
    visualizer.plot_box_numeric_vs_income('age')
    visualizer.plot_pie_income()

    # -------------------
    # Flow 1: LabelEncoder
    # -------------------
    print("\n=== FLOW 1: LabelEncoder ===")
    pre_train = PreProcessing(df_train)
    df_train_clean = pre_train.clean_data()
    df_train_clean = pre_train.encode_labels('income', fit_encoder=True)
    X_train, y_train = pre_train.process(use_onehot=False)

    pre_test = PreProcessing(df_test, fit_encoder=False)
    pre_test.label_encoder = pre_train.label_encoder
    df_test_clean = pre_test.clean_data()
    df_test_clean = pre_test.encode_labels('income', fit_encoder=False)
    X_test, y_test = pre_test.process(use_onehot=False)

    print(f">>> Train: X={X_train.shape}, y={y_train.shape}")
    print(f">>> Test: X={X_test.shape}, y={y_test.shape}")

    # Huấn luyện mô hình
    tree = DecisionTree(criterion='entropy', max_depth=12, min_samples_split=4, min_samples_leaf=12)
    tree.train(X_train, y_train)
    knn = KNN(n_neighbors=11, p=2, weights='uniform')
    knn.train(X_train, y_train)
    nb = Naive_Bayes_Model()
    nb.train(X_train, y_train)
    rf = RandomForest_Model(
        n_estimators=200, max_depth=None, min_samples_split=10,
        min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=42
    )
    rf.train(X_train, y_train)

    # Đánh giá
    for name, model in [("Decision Tree", tree.model),
                        ("KNN", knn.model),
                        ("Naive Bayes", nb.model),
                        ("Random Forest", rf.model)]:
        evaluator = Evaluator(model, X_test, y_test)
        results = evaluator.evaluate()
        print(f"\n>>> Kết quả đánh giá {name}:")
        for metric, value in results.items():
            print(f"- {metric}: {value:.4f}")

    # Feature importance
    top_n = 15
    feature_names = list(X_train.columns)
    importances = tree.model.feature_importances_
    sorted_idx = importances.argsort()[-top_n:]
    print("\nTop feature names & importances:")
    for i in sorted_idx:
        print(f"- {feature_names[i]}: {importances[i]:.4f}")
    plotter = Plotter(tree.model, X_test, y_test)
    plotter.feature_importance(feature_names, top_n=top_n)

    # -------------------
    # Flow 2: OneHotEncoder
    # -------------------
    print("\n=== FLOW 2: OneHotEncoder ===")

    # Tiền xử lý OneHotEncoder
    pre_train_oh = PreProcessing(df_train)
    df_train_clean = pre_train_oh.clean_data()
    X_train_oh, y_train_oh = pre_train_oh.process(use_onehot=True)

    pre_test_oh = PreProcessing(df_test, fit_encoder=False)
    pre_test_oh.transformer = pre_train_oh.transformer
    pre_test_oh.label_encoder = pre_train_oh.label_encoder
    df_test_clean = pre_test_oh.clean_data()
    X_test_oh, y_test_oh = pre_test_oh.process(use_onehot=True)

    print(f">>> Train OH: X={X_train_oh.shape}, y={y_train_oh.shape}")
    print(f">>> Test OH: X={X_test_oh.shape}, y={y_test_oh.shape}")

    # Huấn luyện mô hình
    tree_oh = DecisionTree(criterion='entropy', max_depth=12, min_samples_split=4, min_samples_leaf=12)
    tree_oh.train(X_train_oh, y_train_oh)

    knn_oh = KNN(n_neighbors=11, p=2, weights='uniform')
    knn_oh.train(X_train_oh, y_train_oh)

    nb_oh = Naive_Bayes_Model()
    nb_oh.train(X_train_oh, y_train_oh)

    rf_oh = RandomForest_Model(
        n_estimators=200, max_depth=None, min_samples_split=10,
        min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=42
    )
    rf_oh.train(X_train_oh, y_train_oh)

    # Đánh giá
    for name, model in [("Decision Tree", tree_oh.model),
                        ("KNN", knn_oh.model),
                        ("Naive Bayes", nb_oh.model),
                        ("Random Forest", rf_oh.model)]:
        evaluator = Evaluator(model, X_test_oh, y_test_oh)
        results = evaluator.evaluate()
        print(f"\n>>> Kết quả đánh giá {name} (OH):")
        for metric, value in results.items():
            print(f"- {metric}: {value:.4f}")

    # Feature importance Decision Tree
    top_n = 15
    feature_names_oh = list(X_train_oh.columns)
    importances_oh = tree_oh.model.feature_importances_
    sorted_idx_oh = importances_oh.argsort()[-top_n:]
    print("\nTop feature names & importances (OH):")
    for i in sorted_idx_oh:
        print(f"- {feature_names_oh[i]}: {importances_oh[i]:.4f}")

    plotter_oh = Plotter(tree_oh.model, X_test_oh, y_test_oh)
    plotter_oh.feature_importance(feature_names_oh, top_n=top_n)