"""Eval: happy-share, win-rate vs start, surprise (perplexity-ish), vocab diversity."""
import math, random
import torch
import torch.nn.functional as F
from .pairs import sample_line, sample_batch, ids_to_words
from .judge import Judge
from .train import trigram_tensors


def happy_share(model, openings, corpus, seed=99) -> float:
    judge = Judge()
    i2w = corpus["i2w"]; EOS = corpus["EOS"]
    flat = sample_batch(model, openings, 1, EOS, max_len=12, seed=seed)
    happy = sum(1 for ids in flat if judge.score(ids_to_words(ids, i2w)) > 0)
    return happy / max(len(flat), 1)


def win_rate(cand, base, openings, corpus, seed=123) -> float:
    judge = Judge()
    i2w = corpus["i2w"]; EOS = corpus["EOS"]
    A = sample_batch(cand, openings, 1, EOS, max_len=12, seed=seed * 100003)
    B = sample_batch(base, openings, 1, EOS, max_len=12, seed=seed * 100003 + 999)
    wins = 0; tot = 0
    for a_ids, b_ids in zip(A, B):
        a = ids_to_words(a_ids, i2w); b = ids_to_words(b_ids, i2w)
        sa, sb = judge.score(a), judge.score(b)
        if sa == sb:
            continue
        tot += 1
        if sa > sb:
            wins += 1
    return wins / max(tot, 1)


def surprise_on_heldout(model, corpus) -> float:
    """Mean exp(CE per token) — 'how many words as unsure as choosing between'."""
    X, Y = trigram_tensors(corpus["heldout_enc"], corpus["EOS"], 2)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        ce = F.cross_entropy(logits, Y, reduction="mean").item()
    return math.exp(ce)


def vocab_used(model, openings, corpus, seed=7):
    i2w = corpus["i2w"]; EOS = corpus["EOS"]
    flat = sample_batch(model, openings, 1, EOS, max_len=12, seed=seed)
    seen = set()
    lines = []
    for ids in flat:
        w = ids_to_words(ids, i2w)
        lines.append(" ".join(w))
        seen.update(w)
    return len(seen), lines


def implicit_reward_accuracy(dpo_model, ref_model, pairs) -> float:
    """beta * (logpi - logref) scores winner higher? (video: 99%)."""
    import torch
    ok = 0
    dpo_model.eval(); ref_model.eval()
    with torch.no_grad():
        for p in pairs:
            op = p["opening"]
            # logprob via forward sums
            def lp(m, op_, ids):
                ctx = list(op_); t = 0.0
                for tok in ids:
                    x = torch.tensor([ctx[-2:]], dtype=torch.long)
                    t += torch.log_softmax(m.forward(x)[0], dim=-1)[tok].item()
                    ctx.append(tok)
                return t
            rw = lp(dpo_model, op, p["winner"]) - lp(ref_model, op, p["winner"])
            rl = lp(dpo_model, op, p["loser"]) - lp(ref_model, op, p["loser"])
            if rw > rl:
                ok += 1
    return ok / max(len(pairs), 1)
