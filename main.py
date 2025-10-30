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
    # Bỏ dòng đầu nếu là header
    df = pd.read_csv(file_path, header=None, skipinitialspace=True, encoding='utf-8', comment='|')
    # Loại bỏ dấu chấm cuối nhãn
    df[14] = df[14].str.strip().str.replace('.', '', regex=False)
    return df

if __name__ == "__main__":
    # I Đọc dữ liệu
    df_train = pd.read_csv("data/adult.data", header=None, skipinitialspace=True, encoding='utf-8')
    
    visualizer = DataVisualizer(df_train)

    # 2 histogram numeric
    visualizer.plot_numeric_pair()

    # 2 barplot categorical
    visualizer.plot_categorical_pair(top_n=6)

    # So sánh income vs 1 feature categorical
    visualizer.plot_income_vs_feature('workclass')

    # Boxplot numeric vs income
    visualizer.plot_box_numeric_vs_income('age')

    # Pie chart income distribution
    visualizer.plot_pie_income()


    df_test = load_adult_test("data/adult.test")


    # II Tiền xử lý
    pre_train = PreProcessing(df_train)
    X_train, y_train = pre_train.process()

    pre_test = PreProcessing(df_test, fit_encoder=pre_train.encoder)
    pre_test.label_encoder = pre_train.label_encoder  
    pre_test.transformer = pre_train.transformer      
    X_test, y_test = pre_test.process()
    
    print(f">>> Train: X={X_train.shape}, y={y_train.shape}")
    print(f">>> Test: X={X_test.shape}, y={y_test.shape}")

    # III Huấn luyện mô hình 
    tree = DecisionTree(criterion='entropy', max_depth=12, min_samples_split=4, min_samples_leaf=12)
    tree.train(X_train, y_train)
    
    knn = KNN(n_neighbors=11, p=2, weights='uniform')
    knn.train(X_train, y_train)
    
    nb = Naive_Bayes_Model()
    nb.train(X_train, y_train)
    
    rf = RandomForest_Model(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    rf.train(X_train, y_train)

    # IV Đánh giá mô hình trên test set
    for name, model in [("Decision Tree", tree.model),
                        ("KNN", knn.model),
                        ("Naive Bayes", nb.model),
                        ("Random Forest", rf.model)]:
        evaluator = Evaluator(model, X_test, y_test)
        results = evaluator.evaluate()
        print(f"\n>>> Kết quả đánh giá {name}:")
        for metric, value in results.items():
            print(f"- {metric}: {value:.4f}")

    # V Feature importance Decision Tree
    top_n = 15
    feature_names = list(X_train.columns)
    importances = tree.model.feature_importances_
    sorted_idx = importances.argsort()[-top_n:]
    print("\nTop feature names & importances:")
    for i in sorted_idx:
        print(f"- {feature_names[i]}: {importances[i]:.4f}")

    plotter = Plotter(tree.model, X_test, y_test)
    plotter.feature_importance(feature_names, top_n=top_n)

print("\nThực thi thành công.")
