import pandas as pd

df=pd.read_csv("data/raw/jigsaw/train.csv")
print(df.shape)
print(df.columns.tolist())
print(df.iloc[:, :3].head(2))