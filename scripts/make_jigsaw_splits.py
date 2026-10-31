import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

df=pd.read_csv("data/raw/jigsaw/train.csv")
df= df.drop_duplicates(subset="comment_text")

train, temp=train_test_split(df, test_size=0.3, random_state=42, stratify=df["toxic"])
val, test=train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["toxic"])

train.to_csv("data/processed/jigsaw_train.csv", index=False)
val.to_csv("data/processed/jigsaw_val.csv", index=False)
test.to_csv("data/processed/jigsaw_test.csv", index=False)

for name, part in [("train", train),("val", val), ("test", test)]:
    print(name, len(part), "toxic rate:", round(part["toxic"].mean(),  4))