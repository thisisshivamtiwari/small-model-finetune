"""Load BANKING77 and draw the per-intent training samples every arm shares."""

from pathlib import Path

import pandas as pd

from smft.labels import LABELS

RAW = Path("data/raw")
URLS = {
    "train": "https://huggingface.co/api/datasets/PolyAI/banking77/parquet/default/train/0.parquet",
    "test": "https://huggingface.co/api/datasets/PolyAI/banking77/parquet/default/test/0.parquet",
}


def load_split(split: str, raw: Path = RAW) -> pd.DataFrame:
    """Rows of {id, text, label} with label as the intent name. ids are stable row numbers in the official split."""
    path = raw / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m smft download` first")
    df = pd.read_parquet(path)
    return pd.DataFrame(
        {
            "id": [f"{split}-{i:05d}" for i in range(len(df))],
            "text": df["text"].str.strip(),
            "label": [LABELS[i] for i in df["label"]],
        }
    )


def sample_per_class(df: pd.DataFrame, k: int | None, seed: int = 0) -> pd.DataFrame:
    """k examples per intent (all of them when k is None), in a seeded shuffled order.

    Smaller samples are nested in larger ones (k=8 is the first 8 of k=32's), so the learning curve adds data rather
    than swapping it.
    """
    shuffled = df.sample(frac=1.0, random_state=seed)
    picked = shuffled if k is None else shuffled.groupby("label", sort=False).head(k)
    return picked.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def split_valid(train: pd.DataFrame, per_class: int = 2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a small per-intent validation set (LoRA early checks) before any training sample is drawn.

    Every arm trains on the remaining pool, so no arm ever sees the validation or test messages.
    """
    valid = train.sample(frac=1.0, random_state=seed + 100).groupby("label", sort=False).head(per_class)
    pool = train[~train["id"].isin(valid["id"])].reset_index(drop=True)
    return pool, valid.reset_index(drop=True)


def shots(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One example per intent for few-shot prompts, in label order. Drawn from the k=8 sample so it is training data."""
    first = sample_per_class(df, 8, seed).groupby("label", sort=False).head(1)
    return first.set_index("label").loc[LABELS].reset_index()[["id", "text", "label"]]
