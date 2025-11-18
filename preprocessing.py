import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PreProcessing:
    def __init__(self, file_path=None, is_test=False, df=None):
        self.is_test = is_test
        self.encoders = {}

        if df is not None:
            self.df = df.copy()
        elif file_path is not None:
            if is_test:
                self.df = pd.read_csv(
                    file_path,
                    header=None,
                    skiprows=1,  # bỏ dòng đầu tiên
                    skipinitialspace=True,
                    comment="|",
                    encoding="utf-8",
                )
            else:
                self.df = pd.read_csv(
                    file_path, header=None, skipinitialspace=True, encoding="utf-8"
                )
        else:
            self.df = None

    def run(self):
        cols = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        self.df.columns = cols

        # Xóa khoảng trắng
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].str.strip()
        if self.is_test:
            self.df["income"] = self.df["income"].str.replace(".", "", regex=False)

        # Xóa dòng có "?"
        self.df = self.df.replace("?", pd.NA).dropna()

        # LabelEncoder tất cả cột chữ
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le

        # income thành 0/1
        self.df["income"] = self.df["income"].map({0: 0, 1: 1})

        X = self.df.drop("income", axis=1)
        y = self.df["income"]
        return X, y
