"""TF-IDF + logistic regression: the classic baseline every LLM arm has to beat. Deterministic, runs in CI."""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from smft.data import sample_per_class


def build():
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
    )
    return make_pipeline(features, LogisticRegression(C=10.0, max_iter=3000))


def run(train, test, k: int | None, seed: int) -> list[dict]:
    sample = sample_per_class(train, k, seed)
    model = build()
    t0 = time.perf_counter()
    model.fit(sample["text"], sample["label"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    predicted = model.predict(test["text"])
    per_item_ms = 1000 * (time.perf_counter() - t0) / len(test)
    return [
        {
            "id": row.id,
            "gold": row.label,
            "reply": pred,
            "pred": pred,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round(per_item_ms, 3),
            "train_examples": len(sample),
            "train_seconds": round(fit_s, 2),
        }
        for row, pred in zip(test.itertuples(), predicted, strict=True)
    ]
