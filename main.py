import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from preprocessing import PreProcessing
from models.dt import DecisionTree
from models.knn import KNN 
from evaluate import Evaluator
from plot import Plotter

sys.stdout.reconfigure(encoding='utf-8')

# --- CHẠY CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # 1️⃣ Đọc dữ liệu
    df = pd.read_csv("data/adult.data", header=None, skipinitialspace=True, encoding='utf-8')


    # 2️⃣ Tiền xử lý
    pre = PreProcessing(df)
    X, y = pre.process() 
    print(f">>> Dữ liệu sau tiền xử lý: X={X.shape}, y={y.shape}")

    # 3️⃣ Chia dữ liệu huấn luyện / kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4️⃣ Huấn luyện mô hình 
    tree = DecisionTree(max_depth=6)
    tree.train(X_train, y_train)
    print("✅ Huấn luyện xong DecisionTreeClassifier.")

    knn = KNN(n_neighbors=5, weights='distance')
    knn.train(X_train, y_train)
    print("✅ Huấn luyện xong KNeighborsClassifier.")

    # 5️⃣ Đánh giá mô hình
    evaluator = Evaluator(tree.model, X_test, y_test)
    results = evaluator.evaluate()
    print("\n>>> Kết quả đánh giá mô hình Decision Tree:")
    for metric, value in results.items():
        print(f"- {metric}: {value:.4f}")


    evaluator_knn = Evaluator(knn.model, X_test, y_test)
    results_knn = evaluator_knn.evaluate()
    print("\n>>> Kết quả đánh giá KNN:")
    for metric, value in results_knn.items():
        print(f"- {metric}: {value:.4f}")

    # 6️⃣ Top feature quan trọng
    top_n = 15
    feature_names = list(X.columns)  # convert sang list để plot đẹp
    importances = tree.model.feature_importances_
    sorted_idx = importances.argsort()[-top_n:]
    print("\nTop feature names & importances:")
    for i in sorted_idx:
        print(f"- {feature_names[i]}: {importances[i]:.4f}")

    # 7️⃣ Vẽ biểu đồ feature importance
    plotter = Plotter(tree.model, X_test, y_test)
    plotter.feature_importance(feature_names, top_n=top_n)




# Đây là tên các cột trong DataFrame đã xử lý, bao gồm:

# Numeric columns: fnlwgt, age, hours-per-week, capital-gain, capital-loss, education-num

# One-hot columns: native-country_X, workclass_X, occupation_X, relationship_X, education_X, marital-status_X

# Với one-hot, mỗi giá trị riêng biệt trong cột categorical trở thành một feature. Ví dụ:

# workclass_Private = 1 nếu người đó làm Private, 0 nếu không.

# marital-status_Married-civ-spouse = 1 nếu người đó married, 0 nếu không.