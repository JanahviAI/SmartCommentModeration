from datasets import load_dataset

df= load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train").to_pandas()
spread=df.groupby("comment_id")["hate_speech_score"].std()
print(spread.describe())