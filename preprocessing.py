import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

class PreProcessing:
    def __init__(self, df, fit_encoder=True):
        self.df = df.copy()
        self.fit_encoder = fit_encoder  # True: fit, False: transform only
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

        if self.fit_encoder:
            self.label_encoder = LabelEncoder()
            self.df['income'] = self.label_encoder.fit_transform(self.df['income'])
        else:
            self.df['income'] = self.label_encoder.transform(self.df['income'])

        return self.df

    def process(self, fit_encoder=None):
        if fit_encoder is not None:
            self.fit_encoder = fit_encoder

        self.clean_data()
        X = self.df.drop('income', axis=1)
        y = self.df['income']

        categorical_cols = X.select_dtypes(include='object').columns
        numeric_cols = X.select_dtypes(exclude='object').columns

        if self.fit_encoder:
            self.transformer = ColumnTransformer([
                ('num', MinMaxScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])
            X_processed = self.transformer.fit_transform(X)
        else:
            X_processed = self.transformer.transform(X)

        # Lấy tên cột
        cat_feature_names = []
        if 'cat' in self.transformer.named_transformers_:
            cat_feature_names = self.transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_columns = list(numeric_cols) + list(cat_feature_names)
        X_df = pd.DataFrame(X_processed, columns=all_columns)

        return X_df, y
