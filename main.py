import pandas as pd
from preprocessing import PreProcessing

temp = pd.read_csv("data/adult.data")
df = temp[:10]
print(df.head())
prep = PreProcessing(df)
X, y = prep.process()

print(X.head())
print(y.unique())
