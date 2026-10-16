"""Live public data: Gutenberg Shakespeare #100 -> 22k lines, 4k vocab.

Fully reproducible from the live web (no vendored corpus):
  1. download https://www.gutenberg.org/cache/epub/100/pg100.txt (Complete Works, public domain)
  2. split into lines, tokenize [a-z'] words, lowercase
  3. keep lines with 3..14 tokens (so a 2-word opening + 1..12 continuation)
  4. subsample deterministically to 22,000 lines (seeded shuffle)
  5. vocab = top 4000 + specials: ?=UNK (rare), <END>=EOS
  6. held-out 2000 real lines (never used in LM/SFT/DPO) for surprise eval

If download fails (offline), fall back to a seeded synthetic Shakespeare-like
generator so the pipeline still runs — but RESULTS.json records the source.
"""
import re, random, urllib.request, hashlib
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

UNK_TOKEN = "?"
EOS_TOKEN = "<END>"

GUTENBERG_FALLBACK_LINES = 22000


def fetch_gutenberg(url: str, cache: Path) -> str:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 500_000:
        return cache.read_text(encoding="utf-8", errors="ignore")
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8", errors="ignore")
    cache.write_text(text, encoding="utf-8")
    return text


def synthetic_lines(n: int, seed: int = 0) -> List[List[str]]:
    rnd = random.Random(seed)
    base = ("the king and queen love sweet friend joy fair gentle kind hope grace "
            "glad merry bless dear honour noble true welcome happy death grief blood "
            "sorrow hate fear woe curse grave tears pain sad die kill murder poison war hell "
            "thou thee thy doth hath ere whence hence art wilt shall will not i am content to "
            "do truth for had myself just one that go would see it so wild roth is it a of me").split()
    lines = []
    for _ in range(n):
        L = rnd.randint(4, 12)
        lines.append([rnd.choice(base) for _ in range(L)])
    return lines


def build_corpus(url: str, cache: Path, n_lines: int = 22000,
                 vocab_keep: int = 4000, seed: int = 0,
                 heldout_n: int = 2000) -> Dict:
    try:
        text = fetch_gutenberg(url, cache)
        # strip Gutenberg header/footer heuristically
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        body = text[start:end] if (start >= 0 and end > start) else text
        raw_lines = [l.strip().lower() for l in body.splitlines()]
        toks = []
        for l in raw_lines:
            w = WORD.findall(l)
            if 3 <= len(w) <= 14:
                toks.append(w)
        rnd = random.Random(seed)
        rnd.shuffle(toks)
        # reserve held-out first
        heldout = toks[:heldout_n]
        pool = toks[heldout_n:]
        # ensure enough; wrap if short (shouldn't be — Complete Works is huge)
        while len(pool) < n_lines:
            pool = pool + pool
        lines = pool[:n_lines]
        source = "gutenberg-live"
        sha = hashlib.sha256("".join(" ".join(l) for l in lines[:500]).encode()).hexdigest()[:16]
    except Exception as e:
        lines = synthetic_lines(n_lines, seed)
        heldout = synthetic_lines(heldout_n, seed + 999)
        source = f"synthetic-fallback({type(e).__name__})"
        sha = "synthetic"
    counts = Counter(w for l in lines for w in l)
    keep = set(w for w, _ in counts.most_common(vocab_keep))
    # word->id: 0..3999 vocab, 4000 UNK '?', 4001 EOS '<END>'
    vocab_words = sorted(keep)
    w2i = {w: i for i, w in enumerate(vocab_words)}
    UNK = vocab_keep
    EOS = vocab_keep + 1
    V = vocab_keep + 2
    i2w = vocab_words + [UNK_TOKEN, EOS_TOKEN]

    def encode_line(words: List[str]) -> List[int]:
        return [w2i.get(w, UNK) for w in words]

    encoded = [encode_line(l) for l in lines]
    heldout_enc = [encode_line(l) for l in heldout]
    return {
        "lines": lines, "encoded": encoded,
        "heldout": heldout, "heldout_enc": heldout_enc,
        "w2i": w2i, "i2w": i2w, "UNK": UNK, "EOS": EOS, "V": V,
        "source": source, "sha": sha,
    }
