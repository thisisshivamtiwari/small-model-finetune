import json

import pytest

from smft import prompts
from smft.data import load_split, sample_per_class, shots, split_valid
from smft.labels import LABELS
from smft.metrics import paired_accuracy_diff, score
from smft.parse import parse_label


@pytest.fixture(scope="module")
def splits():
    train, test = load_split("train"), load_split("test")
    pool, valid = split_valid(train)
    return pool, valid, test


def test_official_split_sizes(splits):
    pool, valid, test = splits
    assert len(pool) + len(valid) == 10_003
    assert len(test) == 3_080 and test["label"].value_counts().eq(40).all()
    assert set(test["label"]) == set(LABELS)


def test_no_overlap_between_pool_valid_and_test(splits):
    pool, valid, test = splits
    assert not set(pool["id"]) & set(valid["id"])
    assert valid["label"].value_counts().eq(2).all()
    # A few BANKING77 texts appear in both official splits; count them so the README can say so.
    assert len(set(pool["text"].str.lower()) & set(test["text"].str.lower())) < 30


def test_samples_are_nested_and_balanced(splits):
    pool, _, _ = splits
    k8, k32 = sample_per_class(pool, 8), sample_per_class(pool, 32)
    assert k8["label"].value_counts().eq(8).all()
    assert k32["label"].value_counts().eq(32).all()
    assert set(k8["id"]) <= set(k32["id"])
    assert len(sample_per_class(pool, None)) == len(pool)
    assert sample_per_class(pool, 8).equals(k8)


def test_shots_are_one_per_intent_from_training_sample(splits):
    pool, _, _ = splits
    s = shots(pool)
    assert list(s["label"]) == LABELS
    assert set(s["id"]) <= set(sample_per_class(pool, 8)["id"])


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("card_arrival", "card_arrival"),
        ("  Card_Arrival.\n", "card_arrival"),
        ("`card arrival`", "card_arrival"),
        ("Label: card_arrival", "card_arrival"),
        ("reverted_card_payment?", "reverted_card_payment"),
        ("The intent is card_arrival.", "card_arrival"),
        ("card_arrivals", None),
        ("card", None),
        ("", None),
        ("card_arrival or card_linking", None),
    ],
)
def test_parse_label(reply, expected):
    assert parse_label(reply) == expected


def test_parse_prefers_full_label_over_its_prefix():
    # "exchange_rate" is a prefix of no label but a substring of two; only the exact name should match.
    assert parse_label("card_payment_wrong_exchange_rate") == "card_payment_wrong_exchange_rate"


def test_prompts():
    assert "card_arrival" not in prompts.FINETUNED_SYSTEM
    assert all(label in prompts.LABELED_SYSTEM for label in LABELS)
    few = prompts.few_shot_messages("hi", [("where is my card", "card_arrival")])
    assert [m["role"] for m in few] == ["system", "user", "assistant", "user"]
    record = prompts.training_record("where is my card", "card_arrival")
    assert record["messages"][-1] == {"role": "assistant", "content": "card_arrival"}
    json.dumps(record)


def test_score_counts_invalid_as_wrong():
    gold = ["card_arrival", "card_linking", "age_limit", "age_limit"]
    pred = ["card_arrival", None, "age_limit", "card_linking"]
    s = score(gold, pred, n_boot=50)
    assert s["accuracy"] == 0.5 and s["invalid_rate"] == 0.25
    assert s["accuracy_ci95"][0] <= 0.5 <= s["accuracy_ci95"][1]


def test_paired_diff_sign():
    gold = ["age_limit"] * 10
    better = ["age_limit"] * 10
    worse = ["age_limit"] * 5 + ["card_linking"] * 5
    d = paired_accuracy_diff(gold, better, worse, n_boot=200)
    assert d["diff"] == 0.5 and d["ci95"][0] > 0 and d["p_a_better"] >= 0.99
