"""Fixed rule judge: happy minus sad word counts.

Video gives examples only (happy: love/sweet/friend; sad: death/grief/blood)
and says 18 of each. We fix an explicit 18+18 Shakespeare-plausible lexicon
so the experiment is reproducible. All words are lower-cased, in-vocab.
"""
from typing import List

HAPPY_WORDS: List[str] = [
    "love", "sweet", "friend", "joy", "fair", "gentle", "kind",
    "hope", "grace", "glad", "merry", "bless", "dear", "honour",
    "noble", "true", "welcome", "happy",
]

SAD_WORDS: List[str] = [
    "death", "grief", "blood", "sorrow", "hate", "fear", "woe",
    "curse", "grave", "tears", "pain", "sad", "die", "kill",
    "murder", "poison", "war", "hell",
]

assert len(HAPPY_WORDS) == 18
assert len(SAD_WORDS) == 18

HAPPY = set(HAPPY_WORDS)
SAD = set(SAD_WORDS)


class Judge:
    def score(self, words: List[str]) -> int:
        s = 0
        for w in words:
            wl = w.lower()
            if wl in HAPPY:
                s += 1
            elif wl in SAD:
                s -= 1
        return s

    def pick(self, a: List[str], b: List[str]):
        """Return (winner_idx, score_a, score_b) or (None, s, s) on tie."""
        sa, sb = self.score(a), self.score(b)
        if sa > sb:
            return 0, sa, sb
        if sb > sa:
            return 1, sa, sb
        return None, sa, sb
