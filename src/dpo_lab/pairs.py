"""Sample openings + line pairs from a LM, label with Judge."""
import random
import torch
from typing import List, Dict
from .judge import Judge


def sample_line(model, opening_ids: List[int], w2i, i2w, UNK, EOS,
                rnd: random.Random, max_len: int = 12, temperature: float = 1.0) -> List[int]:
    model.eval()
    ctx = list(opening_ids)
    out = []
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([ctx[-2:]], dtype=torch.long)
            logits = model.forward(x)[0] / temperature
            probs = torch.softmax(logits, dim=-1).numpy()
            nxt = int(rnd.choices(range(len(probs)), weights=probs, k=1)[0])
            out.append(nxt)
            ctx.append(nxt)
            if nxt == EOS:
                break
        else:
            # forced stop not counted; ensure EOS-terminated lines only if model chose EOS
            pass
    return out


def sample_batch(model, openings_list: List[List[int]], n_per_opening: int,
                   EOS: int, max_len: int = 12, temperature: float = 1.0,
                   seed: int = 0, batch_size: int = 4096) -> List[List[int]]:
    """Batched sampling: 1 forward per position for thousands of lines at once.

    Uses torch.multinomial (fast, no Python weight loop). Deterministic given seed.
    Returns flat list len(openings)*n_per_opening of id-lists (incl. EOS if hit).
    """
    import torch
    model.eval()
    g = torch.Generator().manual_seed(seed)
    # expand: each opening repeated n_per_opening
    ctxs = [list(op) for op in openings_list for _ in range(n_per_opening)]
    outs = [[] for _ in ctxs]
    done = [False] * len(ctxs)
    with torch.no_grad():
        for _ in range(max_len):
            active = [i for i, d in enumerate(done) if not d]
            if not active:
                break
            # batch in chunks to bound memory (B x 4002)
            for s in range(0, len(active), batch_size):
                chunk = active[s:s + batch_size]
                x = torch.tensor([ctxs[i][-2:] for i in chunk], dtype=torch.long)
                logits = model.forward(x) / temperature
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=g).squeeze(1).tolist()
                if isinstance(nxt, int):
                    nxt = [nxt]
                for ii, tok in zip(chunk, nxt):
                    outs[ii].append(int(tok))
                    ctxs[ii].append(int(tok))
                    if int(tok) == EOS:
                        done[ii] = True
    return outs


def ids_to_words(ids: List[int], i2w) -> List[str]:
    # drop trailing EOS for scoring/display; keep '?' as-is
    return [i2w[i] for i in ids if i2w[i] != "<END>"]


def make_pairs(model, corpus: Dict, n_openings: int, seed: int = 1) -> Dict:
    rnd = random.Random(seed)
    judge = Judge()
    enc = corpus["encoded"]
    i2w = corpus["i2w"]
    UNK, EOS = corpus["UNK"], corpus["EOS"]
    # openings: first 2 ids of random corpus lines that are in-vocab (not UNK)
    openings = []
    cands = [l for l in enc if len(l) >= 4]
    rnd.shuffle(cands)
    for l in cands:
        if len(openings) >= n_openings:
            break
        op = l[:2]
        if UNK in op:
            continue
        openings.append(op)
    # batched: 2 lines per opening in one go
    flat = sample_batch(model, openings, 2, EOS, max_len=12, seed=seed + 11)
    pairs = []
    ties = 0
    for k, op in enumerate(openings):
        a, b = flat[2 * k], flat[2 * k + 1]
        wa, wb = ids_to_words(a, i2w), ids_to_words(b, i2w)
        pick, sa, sb = judge.pick(wa, wb)
        if pick is None:
            ties += 1
            continue
        if pick == 0:
            w, l = a, b
        else:
            w, l = b, a
        pairs.append({"opening": op, "winner": w, "loser": l,
                      "score_w": max(sa, sb), "score_l": min(sa, sb)})
    return {"pairs": pairs, "ties": ties, "n_openings": len(openings),
            "openings": openings}
