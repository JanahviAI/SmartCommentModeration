from data.loaders import load_measuring_hate

df=load_measuring_hate()
print("Shape:", df.shape)
print("Hate rate:", round(df["hate"].mean(), 3))
print("Identity-targeted rate:", round(df["target_race"].mean(),3 ))
print(df[["comment_id", "hate_speech_score", "hate", "text"]].head(3))
