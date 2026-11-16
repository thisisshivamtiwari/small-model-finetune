"""Chat prompts for each arm. The fine-tuned arm's prompt has no label list: the adapter has learned the labels."""

from smft.labels import LABELS

TASK = "You route online-banking customer messages to one support intent."

FINETUNED_SYSTEM = f"{TASK} Reply with the intent label only."

LABELED_SYSTEM = f"{TASK} Reply with exactly one label from this list and nothing else:\n" + "\n".join(LABELS)


def finetuned_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": FINETUNED_SYSTEM}, {"role": "user", "content": text}]


def zero_shot_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": LABELED_SYSTEM}, {"role": "user", "content": text}]


def few_shot_messages(text: str, examples: list[tuple[str, str]]) -> list[dict]:
    """examples: (text, label) pairs, shown as earlier user/assistant turns so the reply format is learned in context."""
    messages = [{"role": "system", "content": LABELED_SYSTEM}]
    for example_text, label in examples:
        messages += [{"role": "user", "content": example_text}, {"role": "assistant", "content": label}]
    return messages + [{"role": "user", "content": text}]


def training_record(text: str, label: str) -> dict:
    return {"messages": finetuned_messages(text) + [{"role": "assistant", "content": label}]}
