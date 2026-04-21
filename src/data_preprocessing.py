import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_and_process():
    fake = pd.read_csv("data/raw/Fake.csv")
    real = pd.read_csv("data/raw/True.csv")

    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real])
    df = df.sample(frac=1).reset_index(drop=True)

    df["text"] = df["text"].apply(clean_text)

    return df