import os
from sklearn.model_selection import train_test_split
from data.loaders import load_measuring_hate

os.makedirs("data/processed", exist_ok=True)

df = load_measuring_hate()
df["toxic"] = df["hate"]  # use the same column name as Jigsaw for the stratified split

train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["toxic"])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["toxic"])

train.to_csv("data/processed/hate_train.csv", index=False)
val.to_csv("data/processed/hate_val.csv", index=False)
test.to_csv("data/processed/hate_test.csv", index=False)

for name, part in [("train", train), ("val", val), ("test", test)]:
    print(name, len(part), "hate rate:", round(part["toxic"].mean(), 4))