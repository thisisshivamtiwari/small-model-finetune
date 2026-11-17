# STUDY: small-model-finetune

Notes for interviews. Numbers come from `results/scores.json`, `results/headline.json` and `results/train_*.json`.

## 1. Problem in one paragraph

Support teams at banks and fintechs route each customer message to one of many fine-grained intents. Prompting a big
general model with the whole intent list is the quick start, but it pays for the list on every message and is often
not accurate enough on overlapping intents. This project measures, on one public benchmark and one laptop, where a
LoRA fine-tune of a 0.5B model lands against prompting bigger models, and against a classic TF-IDF classifier.

## 2. Design choices (and why)

- **BANKING77 instead of hotel reviews.** The backlog first said hotel review aspects, but there is no public hotel
  aspect set with clean, fine-grained labels and a fixed split. BANKING77 is human-written, has 77 overlapping
  intents, a fixed test split of 40 per intent, a CC BY 4.0 licence and published baselines, so the numbers can be
  sanity-checked against the literature.
- **Everything local (MLX, 4-bit Qwen2.5).** The Gemini key isn't available, and a local run makes latency and
  tokens measurable without API noise. The trade-off: "the bigger model" here is 3B, not a frontier model.
- **Same test set for every arm, paired bootstrap.** All arms answer the same 3,080 messages, so differences are
  compared message by message (paired), which gives tighter, fairer intervals than comparing two separate CIs.
- **Nested training samples.** k=8 sits inside k=32, which sits inside the full pool, so the learning curve shows
  the effect of more data, not of a luckier draw.
- **Validation held out before sampling.** 2 per intent (154 messages) never enter any training sample; LoRA
  epochs were chosen on their loss. The test set was never used for a choice.
- **Loss on the label only (`--mask-prompt`).** The model only has to learn the answer, not to reproduce the
  customer's message.
- **No fuzzy label matching.** A made-up label ("get_spare_card") is counted as invalid and wrong. Fuzzy matching
  would flatter the prompted arms, which invent labels most.
- **Prefix KV cache for prompted arms.** The label list (and 77 shots) is prefilled once and reused. Latency is then
  fair to the prompted arms (an API's prompt cache does the same); token counts still show the full billed prompt.
- **CI checks committed predictions.** GitHub's Linux runners can't run MLX, so CI reruns the deterministic
  baseline, re-scores every prediction file, and fails if a number in `results/` or the README drifts.

## 3. Results to remember

| | Accuracy | Prompt tokens per message | Median latency |
|---|---|---|---|
| TF-IDF + logistic regression, all data | 91.4% (90.4–92.3) | 0 | batch, ~0.1 ms |
| LoRA 0.5B, all data (1 epoch, 23 min) | 85.1% (83.9–86.4) | 45 | 95 ms |
| LoRA 0.5B, 8 per intent (616 examples) | 61.6% | 45 | 90 ms |
| Qwen2.5-3B, 77-shot | 48.7% | 2,444 | 446 ms |
| Qwen2.5-3B, zero-shot | 47.9% | 433 | 353 ms |
| Qwen2.5-0.5B, zero-shot | 19.7% (26% invalid labels) | 433 | 66 ms |

- Fine-tuned 0.5B beat 77-shot 3B by 36.5 points (paired 95% CI 34.5 to 38.6); 8 per intent was already +12.9.
- TF-IDF beat the LoRA at every size: −9.1 (k=8), −14.7 (k=32), −6.2 (all) points.
- Validation loss guided epochs: k=8 at 4 epochs ended at 0.545 and still falling; at 8 epochs, 0.222.

## 4. Failure modes seen

- **GPU out of memory at batch 16** on 8 GB, 80% through the first LoRA run. Fixed with batch 8 plus 2-step
  gradient accumulation and gradient checkpointing (same effective batch).
- **Invented labels.** Prompted models answer with plausible names that aren't in the list (`get_virtual_card`,
  `source_of_funds`, `top_up_charge`). The fine-tuned k=8 model still did it 4.5% of the time
  (`wrong_exchange_rate_for_statement`); with all data, 0.2%.
- **Label conventions.** `get_physical_card` messages are all about PINs, so a model reading label names maps them
  to `passcode_forgotten`. `transfer_timing` vs `transfer_not_received_by_recipient` is the LoRA's top confusion
  (30 messages): the boundary is in the annotation guide, not the words.
- **Few-shot didn't scale.** 77 shots added 2,000 tokens and 0.8 points. One example per intent can't show a
  boundary between two near-identical intents.
- **Undertrained runs look like "small data is bad".** The first k=8 run (4 epochs) was undertrained; reading the
  validation curve, not the test score, caught it.

## 5. Ten likely interview questions

1. **Why did TF-IDF beat a fine-tuned LLM?** The messages are short and keyword-driven, with ~130 labelled examples
   per intent; n-grams plus a linear model fit that well and train in seconds. Our LoRA was small (rank 8, 0.5B,
   1–8 epochs, sized for 8 GB). A larger base or longer training should close some of the gap; the point is to
   measure the cheap baseline first.
2. **Then why fine-tune at all?** Against prompting it wins clearly: +36.5 points with a 10x shorter prompt than
   zero-shot. LLM fine-tunes also handle long, messy or multilingual text better than n-grams, and can return
   structured output; this dataset doesn't test that.
3. **Why paired bootstrap?** Every arm answers the same messages, so the per-message differences remove the shared
   difficulty. Its interval is much tighter than comparing two independent CIs.
4. **How did you avoid test leakage?** Validation (2 per intent) is held out before sampling; epochs were chosen
   on validation loss; the test set was used only for the final score. 6 test texts also appear in train in the
   official split; I left them in, as published.
5. **What does `--mask-prompt` do and why?** The loss is computed only on the answer tokens, so the model doesn't
   spend capacity modelling the customer text or system prompt.
6. **How did you make latency fair to the prompted models?** The fixed prefix (label list, shots) is prefilled once
   into a KV cache and trimmed back after each message, like an API prompt cache. Token counts still bill the full
   prompt.
7. **What would change with a frontier model?** Zero-shot accuracy would likely be much higher, so the break-even
   training size would move up. The token cost per message stays proportional to the label list, which is the
   structural argument for fine-tuning at high volume. I'd measure it rather than assume.
8. **How would you deploy this?** TF-IDF or the LoRA behind a router, with a confidence threshold that sends unsure
   messages to a larger model or a human; log every routing decision and monitor per-intent accuracy drift.
9. **Why QLoRA on a 4-bit base?** The full 0.5B in 16-bit fits, but 4-bit leaves memory for batch and activations
   on 8 GB, and the adapter is only 11 MB (2.9M trainable parameters, 0.6%).
10. **What would you do next?** Three seeds per size, tune rank and epochs, try a 1.5B base, add a frontier arm with
    real cost, and build the hybrid router, measuring accuracy against the share sent to the LLM.
