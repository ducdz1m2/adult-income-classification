import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Plotter:
    def __init__(self, model, X_test, y_test, class_labels=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)

        # Nếu user không truyền class_labels, tự gán 0/1
        if class_labels is None:
            self.class_labels = ["Class 0", "Class 1"]
        else:
            self.class_labels = class_labels

    def confusion_matrix(self):
        """Hiển thị confusion matrix với seaborn."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

    def feature_importance(self, feature_names, top_n=15):
        """Hiển thị top N đặc trưng quan trọng nhất (nếu có)."""
        if not hasattr(self.model, "feature_importances_"):
            print("⚠️ Model này không có thuộc tính feature_importances_.")
            return

        feature_names = list(feature_names)  # <-- bắt buộc
        importances = np.array(self.model.feature_importances_)
        sorted_idx = importances.argsort()[-top_n:]

        plt.figure(figsize=(8, 5))
        plt.barh(np.array(feature_names)[sorted_idx].astype(str), importances[sorted_idx], color='skyblue')
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

