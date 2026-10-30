from datasets import load_dataset

ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")

print(ds)
print(ds["train"].column_names[:20])
print(ds["train"][0])