"""Run one LLM arm over the test set with MLX and write predictions/<arm>.jsonl (resumable, one line per message).

The fixed part of every prompt (system prompt, label list, few-shot turns) is prefilled once into a KV cache and
reused, the way an API's prompt cache would be; each message only pays for its own tokens in latency. Token counts
still report the full prompt, because that is what an API bills.
"""

import json
import time
from pathlib import Path

from smft import prompts
from smft.config import model_path
from smft.data import shots
from smft.parse import parse_label


def messages_fn(arm: dict, pool):
    if arm["kind"] == "zero_shot":
        return prompts.zero_shot_messages
    if arm["kind"] == "finetuned":
        return prompts.finetuned_messages
    if arm["kind"] == "few_shot":
        examples = [(r.text, r.label) for r in shots(pool).itertuples()]
        return lambda text: prompts.few_shot_messages(text, examples)
    raise ValueError(arm["kind"])


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def run(name: str, arm: dict, cfg: dict, test, pool, out: Path, adapter: Path | None = None) -> Path:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    model, tokenizer = load(str(model_path(cfg, arm["model"])), adapter_path=str(adapter) if adapter else None)
    build = messages_fn(arm, pool)

    def encode(text: str) -> list[int]:
        return list(tokenizer.apply_chat_template(build(text), add_generation_prompt=True))

    prefix = encode("a")[: _common_prefix(encode("a"), encode("b"))]
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    for i in range(0, len(prefix), 512):
        model(mx.array(prefix[i : i + 512])[None], cache=cache)
        mx.eval([c.state for c in cache])
    prefill_s = time.perf_counter() - t0

    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"arm": name, **arm, "prefix_tokens": len(prefix), "prefix_prefill_seconds": round(prefill_s, 3)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    with out.open("a") as f:
        for row in test.itertuples():
            if row.id in done:
                continue
            tokens = encode(row.text)
            assert tokens[: len(prefix)] == prefix, "prompt prefix changed; cache would be wrong"
            t0 = time.perf_counter()
            reply, n_out = "", 0
            for response in stream_generate(
                model, tokenizer, mx.array(tokens[len(prefix) :]), max_tokens=cfg["max_tokens"], prompt_cache=cache
            ):
                reply += response.text
                n_out += 1
                if "\n" in reply:
                    break
            latency_ms = 1000 * (time.perf_counter() - t0)
            trim_prompt_cache(cache, cache[0].offset - len(prefix))
            record = {
                "id": row.id,
                "gold": row.label,
                "reply": reply.strip(),
                "pred": parse_label(reply),
                "prompt_tokens": len(tokens),
                "output_tokens": n_out,
                "latency_ms": round(latency_ms, 2),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
    return out
