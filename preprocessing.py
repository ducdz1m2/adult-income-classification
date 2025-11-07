import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

class PreProcessing:
    def __init__(self, df, fit_encoder=True):
        self.df = df.copy()
        self.fit_encoder = fit_encoder
        self.encoder = None
        self.scaler = None
        self.transformer = None
        self.label_encoder = None

    def clean_data(self):
        self.df.columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        self.df = self.df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
        self.df = self.df.replace('?', pd.NA).dropna()
        return self.df

    def encode_labels(self, column, fit_encoder=None):
        """Mã hóa nhãn bằng LabelEncoder"""
        if fit_encoder is not None:
            self.fit_encoder = fit_encoder

        if self.fit_encoder:
            self.label_encoder = LabelEncoder()
            self.df[column] = self.label_encoder.fit_transform(self.df[column])
        else:
            self.df[column] = self.label_encoder.transform(self.df[column])
        return self.df

    def process(self, use_onehot=True, fit_encoder=None):
        if fit_encoder is not None:
            self.fit_encoder = fit_encoder

        X = self.df.drop('income', axis=1)
        y = self.df['income']

        categorical_cols = X.select_dtypes(include='object').columns
        numeric_cols = X.select_dtypes(exclude='object').columns

        if use_onehot:
            if self.fit_encoder:
                self.transformer = ColumnTransformer([
                    ('num', MinMaxScaler(), numeric_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ])
                X_processed = self.transformer.fit_transform(X)
            else:
                X_processed = self.transformer.transform(X)

            cat_feature_names = []
            if 'cat' in self.transformer.named_transformers_:
                cat_feature_names = self.transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            all_columns = list(numeric_cols) + list(cat_feature_names)
            X_df = pd.DataFrame(X_processed, columns=all_columns)
        else:
            # Chỉ giữ numeric + label encoded categorical
            X_df = X.copy()
            for col in categorical_cols:
                X_df[col] = LabelEncoder().fit_transform(X_df[col])
            # Nếu X đã có label encoder trước đó, có thể tái sử dụng
            # nhưng ở đây chỉ demo simple
        return X_df, y
