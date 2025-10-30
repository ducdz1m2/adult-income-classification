from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """Trả về dict chứa 4 chỉ số cơ bản."""
        # Dự đoán nhãn của tập kiểm thử
        y_pred = self.model.predict(self.X_test)

        # Tính toán các chỉ số đánh giá
        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        }

        # In kết quả đánh giá ra màn hình
        print("=== Evaluation Metrics ===")
        for k, v in metrics.items():
            print(f"{k.capitalize():<10}: {v:.4f}")
        print("==========================")

        # Trả về dictionary chứa các chỉ số
        return metrics
