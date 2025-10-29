import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, df):
        self.df = df.copy()
        self.clean_columns()

    def clean_columns(self):
        self.df.columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        self.df = self.df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    def plot_numeric_pair(self, cols=None):
        """Vẽ 2 histogram numeric trên cùng figure"""
        if cols is None:
            cols = self.df.select_dtypes(include='number').columns[:2]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col in zip(axes, cols):
            sns.histplot(self.df[col], bins=30, kde=True, color='skyblue', ax=ax)
            ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

    def plot_categorical_pair(self, cols=None, top_n=5):
        """Vẽ 2 barplot categorical trên cùng figure"""
        if cols is None:
            cols = self.df.select_dtypes(include='object').columns[:2]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col in zip(axes, cols):
            counts = self.df[col].value_counts().head(top_n)
            sns.barplot(x=counts.values, y=counts.index, palette='viridis', ax=ax)
            ax.set_title(f"Top {top_n} categories in {col}")
        plt.tight_layout()
        plt.show()

    def plot_income_vs_feature(self, feature):
        """So sánh income vs categorical feature"""
        plt.figure(figsize=(6, 4))
        sns.countplot(x=feature, hue='income', data=self.df, palette='Set2', order=self.df[feature].value_counts().index)
        plt.title(f"{feature} vs Income")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_pie_income(self):
        """Pie chart % <=50K / >50K"""
        counts = self.df['income'].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['lightcoral','lightskyblue'], startangle=90)
        plt.title("Income distribution")
        plt.tight_layout()
        plt.show()

    def plot_box_numeric_vs_income(self, col):
        """Boxplot numeric theo income"""
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='income', y=col, data=self.df, palette='Set3')
        plt.title(f"{col} by Income")
        plt.tight_layout()
        plt.show()
