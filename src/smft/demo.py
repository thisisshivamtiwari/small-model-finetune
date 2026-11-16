"""A short terminal walkthrough of the committed results, used for the demo video. Reads predictions/ and results/."""

import json
import time

from smft.data import load_split
from smft.report import PRED, RESULTS, load_predictions

SHOW = ["tfidf-lr-all", "zeroshot-qwen3b", "fewshot77-qwen3b", "lora-qwen0.5b-k8", "lora-qwen0.5b-all"]


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def run(pause: float = 1.0) -> None:
    scores = json.loads((RESULTS / "scores.json").read_text())
    head = json.loads((RESULTS / "headline.json").read_text())
    text = dict(zip(*load_split("test")[["id", "text"]].values.T, strict=True))
    preds = {name: {r["id"]: r for r in load_predictions(PRED / f"{name}.jsonl")} for name in SHOW if name in scores}
    print("BANKING77: 3,080 real online-banking support messages, 77 intents, 40 each (public, CC BY 4.0)")
    print("Question: fine-tune a 0.5B model, or prompt a 6x bigger one with the intent list?")
    _pause(pause)

    big, small = "fewshot77-qwen3b", "lora-qwen0.5b-all"
    print("\nMessages the prompted 3B model got wrong and the fine-tuned 0.5B got right:")
    shown = 0
    for record_id, record in preds[small].items():
        other = preds[big][record_id]
        if record["pred"] == record["gold"] and other["pred"] != other["gold"] and shown < 3:
            print(f'  "{text[record_id]}"')
            print(f"     gold {record['gold']:<38} 3B few-shot: {other['reply'][:38]}")
            shown += 1
    _pause(pause)

    print("\nResults on all 3,080 messages (our run, Apple M2 8 GB, 4-bit MLX):")
    print(f"  {'arm':<20} {'accuracy':>9} {'prompt tokens/msg':>18} {'median ms':>10}")
    for name in SHOW:
        if name in scores:
            s = scores[name]
            tokens = s["prompt_tokens_per_1k"] / 1000
            print(f"  {name:<20} {100 * s['accuracy']:>8.1f}% {tokens:>18.0f} {s['latency_ms_median']:>10.0f}")
    _pause(pause)

    pair = head["paired"].get(f"{small} vs {big}")
    if pair:
        lo, hi = (100 * x for x in pair["ci95"])
        print(f"\nfine-tuned 0.5B minus 77-shot 3B: {100 * pair['diff']:+.1f} points (95% CI {lo:+.1f} to {hi:+.1f})")
    base = head["paired"].get(f"{small} vs tfidf-lr-all")
    if base:
        lo, hi = (100 * x for x in base["ci95"])
        print(f"fine-tuned 0.5B minus TF-IDF + logistic regression: {100 * base['diff']:+.1f} points ({lo:+.1f} to {hi:+.1f})")
