"""Turn a model's reply into an intent label, or None when it isn't one of the 77."""

import re

from smft.labels import LABEL_SET, LABELS


def normalise(reply: str) -> str:
    first = reply.strip().splitlines()[0] if reply.strip() else ""
    first = first.strip().strip("`'\".,:;*?!").strip().lower()
    first = re.sub(r"^(label|intent)\s*[:=]\s*", "", first)
    return re.sub(r"[\s-]+", "_", first)


def parse_label(reply: str) -> str | None:
    """Exact label after light clean-up; otherwise the single label named in the reply; otherwise None.

    No fuzzy matching: a near-miss label counts as invalid, so the invalid rate shows format failures honestly.
    """
    cleaned = normalise(reply)
    if cleaned in LABEL_SET:
        return cleaned
    lowered = reply.lower()
    named = [label for label in LABELS if re.search(rf"(?<![a-z_]){re.escape(label)}(?![a-z_])", lowered)]
    return named[0] if len(named) == 1 else None
