import pandas as pd
from datasets import load_dataset

def load_measuring_hate() -> pd.DataFrame:
    """
    Return one row per comment with an average hate score and identity-target flags.
    """
    ds=load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    df=ds.to_pandas()

    target_cols=[c for c in df.columns if c.startswith("target_")]

    agg={
        "text": "first",
        "platform": "first",
        "hate_speech_score": "mean"
    }

    agg.update({c: "max" for c in target_cols})

    out = df.groupby("comment_id").agg(agg).reset_index()
    out["hate"]=(out["hate_speech_score"]>0.5).astype(int)
    return out
