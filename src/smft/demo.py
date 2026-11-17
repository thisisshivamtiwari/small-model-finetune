"""A short terminal walkthrough of the committed results, used for the demo video. Reads predictions/ and results/."""

import json
import time
from collections import Counter

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
    print("BANKING77: 3,080 real banking support messages, 77 intents (public, CC BY 4.0)")
    print("Fine-tune a 0.5B model, or prompt a 6x bigger one with the intent list?")
    _pause(pause)

    big, small = "fewshot77-qwen3b", "lora-qwen0.5b-all"
    print("\nThe 3B model's top mistakes (the fine-tuned 0.5B got each example right):")
    confusions = Counter((r["gold"], r["pred"]) for r in preds[big].values() if r["pred"] is not None and r["pred"] != r["gold"])
    for (gold, wrong), count in confusions.most_common(3):
        example = next(
            i for i, r in preds[big].items() if (r["gold"], r["pred"]) == (gold, wrong) and preds[small][i]["pred"] == gold
        )
        print(f'  "{text[example]}"')
        print(f"     gold: {gold}   3B said: {wrong} ({count}x)")
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
    _pause(pause)
    best = max(scores, key=lambda n: scores[n]["accuracy"])
    print(f"\nBest arm on this run: {best} at {100 * scores[best]['accuracy']:.1f}%.")
    print("Check the boring baseline before paying for a bigger model.")
