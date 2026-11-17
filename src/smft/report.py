"""Score every predictions file, write results/*.json and the README tables between the RESULTS markers."""

import json
import re
from pathlib import Path

import numpy as np

from smft import baseline
from smft.config import k_name
from smft.metrics import paired_accuracy_diff, score

PRED = Path("predictions")
RESULTS = Path("results")
README = Path("README.md")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# Display order: baseline, prompted, fine-tuned.
ORDER = [
    "tfidf-lr-k8", "tfidf-lr-k32", "tfidf-lr-all",
    "zeroshot-qwen0.5b", "zeroshot-qwen1.5b", "zeroshot-qwen3b", "fewshot77-qwen3b",
    "lora-qwen0.5b-k8", "lora-qwen0.5b-k32", "lora-qwen0.5b-all",
]  # fmt: skip
HEADLINE_PAIRS = [
    ("lora-qwen0.5b-all", "fewshot77-qwen3b"),
    ("lora-qwen0.5b-all", "zeroshot-qwen3b"),
    ("lora-qwen0.5b-k8", "fewshot77-qwen3b"),
    ("lora-qwen0.5b-all", "tfidf-lr-all"),
    ("lora-qwen0.5b-k8", "tfidf-lr-k8"),
    ("lora-qwen0.5b-k32", "tfidf-lr-k32"),
]


def load_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _aligned(test, records: list[dict]) -> list[dict] | None:
    by_id = {r["id"]: r for r in records}
    if any(i not in by_id for i in test["id"]):
        return None
    return [by_id[i] for i in test["id"]]


def _train_info(name: str, records: list[dict]) -> dict:
    if name.startswith("tfidf"):
        return {"train_examples": records[0]["train_examples"], "train_seconds": records[0]["train_seconds"]}
    match = re.search(r"-(k\d+|all)$", name)
    if name.startswith("lora") and match and (RESULTS / f"train_{match.group(1)}.json").exists():
        info = json.loads((RESULTS / f"train_{match.group(1)}.json").read_text())
        return {"train_examples": info["train_examples"], "train_seconds": info["train_seconds"]}
    if name.startswith("fewshot"):
        return {"train_examples": 77, "train_seconds": 0}
    return {"train_examples": 0, "train_seconds": 0}


def score_all(cfg: dict, test) -> tuple[dict, dict[str, list]]:
    scores, preds = {}, {}
    for name in ORDER:
        path = PRED / f"{name}.jsonl"
        if not path.exists():
            continue
        records = _aligned(test, load_predictions(path))
        if records is None:
            continue
        gold = [r["gold"] for r in records]
        pred = [r["pred"] for r in records]
        latency = np.array([r["latency_ms"] for r in records])
        scores[name] = {
            **score(gold, pred, cfg["bootstrap"], cfg["seed"]),
            "prompt_tokens_per_1k": int(sum(r["prompt_tokens"] for r in records) * 1000 / len(records)),
            "output_tokens_per_1k": int(sum(r["output_tokens"] for r in records) * 1000 / len(records)),
            "latency_ms_median": round(float(np.median(latency)), 2),
            "latency_ms_p95": round(float(np.percentile(latency, 95)), 2),
            **_train_info(name, records),
        }
        preds[name] = pred
    return scores, preds


def headline(cfg: dict, test, scores: dict, preds: dict) -> dict:
    gold = list(test["label"])
    pairs = {
        f"{a} vs {b}": paired_accuracy_diff(gold, preds[a], preds[b], cfg["bootstrap"], cfg["seed"])
        for a, b in HEADLINE_PAIRS
        if a in preds and b in preds
    }
    prompted = [n for n in scores if n.startswith(("zeroshot", "fewshot"))]
    best_prompted = max(prompted, key=lambda n: scores[n]["accuracy"]) if prompted else None
    break_even = None
    if best_prompted:
        for k in cfg["train_sizes"]:
            name = f"lora-qwen0.5b-{k_name(k)}"
            if name in preds:
                diff = paired_accuracy_diff(gold, preds[name], preds[best_prompted], cfg["bootstrap"], cfg["seed"])
                if diff["ci95"][0] > 0:
                    break_even = {"arm": name, "examples_per_intent": k_name(k), "vs": best_prompted, **diff}
                    break
    return {"best_prompted": best_prompted, "smallest_lora_clearly_better": break_even, "paired": pairs}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def tables(scores: dict, head: dict) -> str:
    lines = [
        "| Arm | Trained on | Accuracy % (95% CI) | Macro-F1 % | Invalid % | Prompt tokens / 1k msgs | Median latency ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in scores.items():
        trained = f"{s['train_examples']:,} examples" if s["train_examples"] else "none"
        if name.startswith("fewshot"):
            trained = "77 shots in prompt"
        lo, hi = s["accuracy_ci95"]
        lines.append(
            f"| {name} | {trained} | {_pct(s['accuracy'])} ({_pct(lo)}–{_pct(hi)}) | {_pct(s['macro_f1'])} | "
            f"{_pct(s['invalid_rate'])} | {s['prompt_tokens_per_1k']:,} | {s['latency_ms_median']:.1f} |"
        )
    if head["paired"]:
        lines += ["", "| Paired comparison (same 3,080 messages) | Accuracy difference, points (95% CI) |", "|---|---|"]
        for pair, d in head["paired"].items():
            lo, hi = d["ci95"]
            lines.append(f"| {pair} | {_pct(d['diff'])} ({_pct(lo)} to {_pct(hi)}) |")
    return "\n".join(lines)


def write_all(cfg: dict, test) -> tuple[dict, dict]:
    scores, preds = score_all(cfg, test)
    head = headline(cfg, test, scores, preds)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "scores.json").write_text(json.dumps(scores, indent=1) + "\n")
    (RESULTS / "headline.json").write_text(json.dumps(head, indent=1) + "\n")
    if README.exists() and START in README.read_text():
        text = README.read_text()
        block = f"{START}\n{tables(scores, head)}\n{END}"
        README.write_text(re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S))
    return scores, head


def ci(cfg: dict, pool, test) -> int:
    """The eval gate: the baseline reproduces, the committed predictions re-score to the committed results, the README
    tables match, and the headline claims still hold."""
    failures = []
    for k in cfg["train_sizes"]:
        name = f"tfidf-lr-{k_name(k)}"
        fresh = [r["pred"] for r in baseline.run(pool, test, k, cfg["seed"])]
        committed = [r["pred"] for r in load_predictions(PRED / f"{name}.jsonl")]
        mismatches = sum(a != b for a, b in zip(fresh, committed, strict=True))
        if mismatches:
            failures.append(f"{name}: {mismatches} predictions differ from a fresh run")
    scores, preds = score_all(cfg, test)
    head = headline(cfg, test, scores, preds)
    if json.loads((RESULTS / "scores.json").read_text()) != json.loads(json.dumps(scores)):
        failures.append("results/scores.json does not match a re-score of predictions/")
    if json.loads((RESULTS / "headline.json").read_text()) != json.loads(json.dumps(head)):
        failures.append("results/headline.json does not match a re-score of predictions/")
    if f"{START}\n{tables(scores, head)}\n{END}" not in README.read_text():
        failures.append("README results tables are stale: run `python -m smft score`")
    # The README's two conclusions; if a rerun flips either, the write-up is wrong and CI must say so.
    pair = head["paired"].get("lora-qwen0.5b-all vs fewshot77-qwen3b")
    if pair is None or pair["ci95"][0] <= 0:
        failures.append("headline no longer holds: full LoRA is not clearly better than 77-shot 3B")
    pair = head["paired"].get("lora-qwen0.5b-all vs tfidf-lr-all")
    if pair is None or pair["ci95"][1] >= 0:
        failures.append("headline no longer holds: TF-IDF is not clearly better than the full LoRA")
    for f in failures:
        print("FAIL", f)
    print("ci: ok" if not failures else f"ci: {len(failures)} failure(s)")
    return 1 if failures else 0
