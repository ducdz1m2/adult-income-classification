from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd

class PreProcessing:
    def __init__(self, df):
        self.df = df.copy()

    def process(self):
        X = self.df.drop(self.df.columns[-1], axis=1)
        y = self.df[self.df.columns[-1]]
        
        categorical_cols = X.select_dtypes(include='object').columns
        numeric_cols = X.select_dtypes(exclude='object').columns
        
        transformer = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            
            # 'num': với các cột số (numeric_cols), áp dụng StandardScaler() → chuẩn hóa về trung bình 0, độ lệch chuẩn 1.
            # 'cat': với các cột chuỗi (categorical_cols), áp dụng OneHotEncoder() → chuyển giá trị chuỗi thành vector nhị phân.
        ])
        
        X_processed = transformer.fit_transform(X)
        
        # Trả lại DataFrame có tên cột rõ ràng
        cat_feature_names = transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_columns = list(numeric_cols) + list(cat_feature_names)
        X_df = pd.DataFrame(X_processed, columns=all_columns)

        return X_df, y
