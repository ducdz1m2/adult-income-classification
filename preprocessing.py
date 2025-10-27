import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# Đảm bảo in Unicode trên Windows
sys.stdout.reconfigure(encoding='utf-8')

class PreProcessing:
    def __init__(self, df):
        self.df = df.copy()

    def inspect_data(self):
        print(">>> Kích thước DataFrame:", self.df.shape)
        print(">>> Các cột trong DataFrame:")
        print(list(self.df.columns))
        print("\n>>> Kiểu dữ liệu:")
        print(self.df.dtypes)
        print("\n>>> 5 dòng đầu:")
        print(self.df.head())
        print("\n>>> Giá trị khác nhau trong từng cột (10 giá trị đầu):")
        for col in self.df.columns:
            print(f"- {col}: {self.df[col].unique()[:10]}")

    def clean_data(self):
        self.df.columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]

        self.df = self.df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
        self.df = self.df.replace('?', pd.NA).dropna()

        # Chỉ mã hóa income
        le = LabelEncoder()
        self.df['income'] = le.fit_transform(self.df['income'])

        return self.df


    def process(self):
        self.clean_data()
        X = self.df.drop('income', axis=1)
        y = self.df['income']

        categorical_cols = X.select_dtypes(include='object').columns
        numeric_cols = X.select_dtypes(exclude='object').columns

        transformer = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

        X_processed = transformer.fit_transform(X)

        cat_feature_names = transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_columns = list(numeric_cols) + list(cat_feature_names)
        X_df = pd.DataFrame(X_processed, columns=all_columns)

        return X_df, y
