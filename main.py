import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from preprocessing import PreProcessing
from models.dt import DecisionTree
from models.knn import KNN 
from models.nb import Naive_Bayes_Model
from models.rf import RandomForest_Model
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
    tree = DecisionTree(criterion= 'gini',max_depth=10,min_samples_split=2, min_samples_leaf=1)
    tree.train(X_train, y_train)
    print("✅ Huấn luyện xong DecisionTreeClassifier.")

    knn = KNN(n_neighbors=11,p=2, weights='uniform')
    knn.train(X_train, y_train)
    print("✅ Huấn luyện xong KNeighborsClassifier.")

    
    nb = Naive_Bayes_Model()
    nb.train(X_train, y_train)
    print("Huấn luyện xong mô hình NaiveBayesClassifier.")

    rf = RandomForest_Model(n_estimators=200, min_samples_split = 10, min_samples_leaf = 1, max_depth=None, bootstrap=True, random_state=42)
    rf.train(X_train, y_train)
    print("Huấn luyện xong mô hình RandomForestClassifier.")

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

    evaluator_nb = Evaluator(nb.model,X_test, y_test)
    results_nb = evaluator_nb.evaluate()
    print("\n Kết quả đánh giá Navie Bayes:")
    for metric, value in results_nb.items():
        print(f"- {metric}: {value:.4f}")

    evaluator_rf = Evaluator(rf.model,X_test, y_test)
    results_rf = evaluator_rf.evaluate()
    print("\n Kết quả đánh giá Random Forests:")
    for metric, value in results_rf.items():
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

# >>> Dữ liệu sau tiền xử lý: X=(30162, 104), y=(30162,)

# === 🔎 Tuning Decision Tree ===
# Best params: {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
# CV F1: 0.6575396211973845
# CV F1: 0.6575396211973845

# === 🔎 Tuning KNN ===
# Best params: {'n_neighbors': 11, 'p': 2, 'weights': 'uniform'}
# CV F1: 0.6328032819678856

# === 🔎 Tuning Naive Bayes ===
# Best params: {'var_smoothing': np.float64(0.1)}
# CV F1: 0.6364977073433581

# === 🔎 Tuning Random Forest ===
# Best params: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': None, 'bootstrap': True}
# CV F1: 0.6907034208522354