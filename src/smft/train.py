"""QLoRA fine-tune with mlx_lm.lora on k examples per intent. Writes adapters/<k>/ and results/train_<k>.json."""

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from smft.config import k_name, model_path
from smft.data import sample_per_class
from smft.prompts import training_record


def _write_jsonl(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in df.itertuples():
            f.write(json.dumps(training_record(row.text, row.label)) + "\n")


def run(cfg: dict, pool, valid, k: int | None, adapters: Path = Path("adapters"), results: Path = Path("results")) -> dict:
    lora = cfg["lora"]
    sample = sample_per_class(pool, k, cfg["seed"])
    out = adapters / k_name(k)
    _write_jsonl(out / "data" / "train.jsonl", sample)
    _write_jsonl(out / "data" / "valid.jsonl", valid)
    epochs = lora["epochs"][k]
    iters = math.ceil(epochs * len(sample) / lora["batch_size"])
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", str(model_path(cfg, lora["base"])),
        "--train", "--data", str(out / "data"), "--adapter-path", str(out),
        "--fine-tune-type", "lora", "--mask-prompt",
        "--num-layers", str(lora["num_layers"]), "--batch-size", str(lora["batch_size"]),
        "--learning-rate", str(lora["learning_rate"]), "--iters", str(iters),
        "--max-seq-length", str(lora["max_seq_length"]), "--seed", str(cfg["seed"]),
        "--steps-per-report", "25", "--steps-per-eval", str(max(iters // 4, 25)), "--val-batches", "-1",
    ]  # fmt: skip
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.perf_counter() - t0
    log = proc.stdout + proc.stderr
    (out / "train.log").write_text(log)
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_lm lora failed:\n{log[-3000:]}")
    train_loss = [float(x) for x in re.findall(r"Train loss ([0-9.]+)", log)]
    val_loss = [float(x) for x in re.findall(r"Val loss ([0-9.]+)", log)]
    peak = [float(x) for x in re.findall(r"Peak mem ([0-9.]+) GB", log)]
    info = {
        "k": k_name(k),
        "base": cfg["models"][lora["base"]],
        "train_examples": len(sample),
        "valid_examples": len(valid),
        "epochs": epochs,
        "iters": iters,
        "batch_size": lora["batch_size"],
        "learning_rate": lora["learning_rate"],
        "num_layers": lora["num_layers"],
        "train_seconds": round(seconds, 1),
        "train_loss_last": train_loss[-1] if train_loss else None,
        "val_loss": val_loss,
        "peak_mem_gb": max(peak) if peak else None,
    }
    results.mkdir(exist_ok=True)
    (results / f"train_{k_name(k)}.json").write_text(json.dumps(info, indent=1) + "\n")
    return info
