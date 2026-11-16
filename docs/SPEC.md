# Spec: when does fine-tuning a small model beat prompting a bigger one?

**Problem.** Banks and fintechs route thousands of support messages a day into fine-grained intents ("card arrived?",
"exchange rate wrong on a cash withdrawal", "top-up reverted"). Teams often prompt a large general model with the full
intent list for every message. That pays for hundreds of prompt tokens per message on a narrow, repetitive task.

**Question.** On a laptop (Apple M2, 8 GB), how many labelled examples does a LoRA fine-tune of a 0.5B model need to
beat zero-shot and few-shot prompting of a 6x larger model, and what does each option cost in tokens and latency?

**Data.** BANKING77 (PolyAI, CC BY 4.0): 13,083 online-banking queries, 77 intents; official split 10,003 train /
3,080 test (40 per intent). Label names come from the dataset card (lower-cased, trailing "?" dropped).

**Arms** (all scored on the same test items):
1. TF-IDF + logistic regression, trained on k = 8, 32 and all examples per intent (classic baseline; reruns in CI).
2. Prompted, zero-shot with the 77 labels in the system prompt: Qwen2.5 0.5B, 1.5B and 3B Instruct (4-bit, MLX).
3. Prompted, few-shot: Qwen2.5-3B with one labelled example per intent (77 shots).
4. Fine-tuned: QLoRA on Qwen2.5-0.5B-Instruct-4bit with k = 8, 32 and all examples per intent; the prompt carries no
   label list.

**Measures.** Accuracy and macro-F1 with bootstrap 95% intervals, paired bootstrap for the headline comparison,
invalid-label rate, prompt and output tokens per 1,000 messages, wall-clock latency per message, training time.

**Rules.** Every number is computed from `predictions/*.jsonl` by `smft score`; CI re-scores the committed predictions
and reruns the TF-IDF baseline. A frontier-API arm (Gemini) waits for the key; nothing about API prices is claimed.
