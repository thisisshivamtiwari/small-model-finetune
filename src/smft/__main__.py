"""CLI: python -m smft <download|baseline|train|infer|score|ci>."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from smft import baseline, report
from smft.config import k_name, load_config
from smft.data import RAW, URLS, load_split, split_valid

PRED = Path("predictions")


def _k(value: str) -> int | None:
    return None if value == "all" else int(value)


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smft")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="fetch BANKING77 parquet files from Hugging Face")
    sub.add_parser("baseline", help="TF-IDF + logistic regression at every training size")
    train = sub.add_parser("train", help="LoRA fine-tune (Apple silicon)")
    train.add_argument("--k", required=True, help="examples per intent, or 'all'")
    infer = sub.add_parser("infer", help="run one LLM arm over the test set (Apple silicon)")
    infer.add_argument("--arm", required=True)
    infer.add_argument("--limit", type=int, help="first N test messages only (smoke test)")
    infer.add_argument("--out", type=Path, help="write predictions here instead of predictions/<arm>.jsonl")
    sub.add_parser("score", help="score every predictions/*.jsonl into results/ and refresh the README tables")
    demo = sub.add_parser("demo", help="short terminal walkthrough of the committed results (for the demo video)")
    demo.add_argument("--pause", type=float, default=1.0)
    sub.add_parser("ci", help="rerun the baseline and re-score; fail if any committed number changed")
    args = parser.parse_args(argv)
    cfg = load_config()

    if args.cmd == "download":
        RAW.mkdir(parents=True, exist_ok=True)
        for split, url in URLS.items():
            urllib.request.urlretrieve(url, RAW / f"{split}.parquet")
        return 0

    test = load_split("test")
    pool, valid = split_valid(load_split("train"), seed=cfg["seed"])

    if args.cmd == "baseline":
        for k in cfg["train_sizes"]:
            _write(PRED / f"tfidf-lr-{k_name(k)}.jsonl", baseline.run(pool, test, k, cfg["seed"]))
        return 0
    if args.cmd == "train":
        from smft import train as trainer

        print(json.dumps(trainer.run(cfg, pool, valid, _k(args.k)), indent=1))
        return 0
    if args.cmd == "infer":
        from smft import infer

        arm = cfg["arms"][args.arm]
        adapter = None
        if arm["kind"] == "finetuned":
            # A fresh training run lands in adapters/; the committed full-data adapter lets a clone skip training.
            adapter = Path("adapters") / k_name(arm["k"])
            if not (adapter / "adapters.safetensors").exists():
                adapter = Path("models") / args.arm
        rows = test.head(args.limit) if args.limit else test
        out = args.out or (PRED / f"{args.arm}.jsonl" if not args.limit else Path("/tmp") / f"smft-smoke-{args.arm}.jsonl")
        if args.limit or args.out:
            out.unlink(missing_ok=True)
        infer.run(args.arm, arm, cfg, rows, pool, out, adapter)
        print(out)
        return 0
    if args.cmd == "score":
        report.write_all(cfg, test)
        return 0
    if args.cmd == "demo":
        from smft import demo as walkthrough

        walkthrough.run(args.pause)
        return 0
    if args.cmd == "ci":
        return report.ci(cfg, pool, test)
    return 1


if __name__ == "__main__":
    sys.exit(main())
