"""Training: LM pre-train, SFT-on-winners, DPO. All CPU, seeded."""
import random, copy, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .model import TinyWordLM


def build_trigrams(encoded, context_len=2):
    X, Y = [], []
    for line in encoded:
        seq = line + [4001]  # EOS id appended (V-1); caller passes real EOS
        # NOTE: generic; we recompute with actual EOS below
        pass
    return X, Y


def trigram_tensors(encoded, EOS, context_len=2):
    Xs, Ys = [], []
    for line in encoded:
        toks = line + [EOS]
        # contexts: start with first `context_len` tokens as context
        ctx = toks[:context_len]
        for nxt in toks[context_len:]:
            Xs.append(list(ctx[-context_len:]))
            Ys.append(nxt)
            ctx.append(nxt)
    return torch.tensor(Xs, dtype=torch.long), torch.tensor(Ys, dtype=torch.long)


def train_lm(corpus, cfg, seed=0):
    torch.manual_seed(seed); random.seed(seed)
    V = corpus["V"]
    model = TinyWordLM(V, cfg.embed_dim, cfg.hidden_dim, cfg.context_len)
    X, Y = trigram_tensors(corpus["encoded"], corpus["EOS"], cfg.context_len)
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=cfg.lm_batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lm_lr)
    model.train()
    for _ in range(cfg.lm_epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def _line_loss(model, opening, line_ids):
    """Cross-entropy sum over line (incl EOS) given opening. Differentiable."""
    ctx = list(opening)
    losses = []
    for tok in line_ids:
        x = torch.tensor([ctx[-2:]], dtype=torch.long)
        logits = model.forward(x)[0]
        losses.append(F.cross_entropy(logits.unsqueeze(0),
                                      torch.tensor([tok])))
        ctx.append(tok)
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).sum()


def _batch_line_logprobs(model, openings, lines):
    """Batched sum-logprob for a batch of (opening, line) pairs. One forward total.

    Builds all (ctx,next) rows, runs one forward, splits sums per line.
    Returns tensor (B,) of sum log P(line|opening), differentiable w.r.t. model.
    """
    rows, owner, lens = [], [], []
    for bi, (op, ln) in enumerate(zip(openings, lines)):
        ctx = list(op)
        for tok in ln:
            rows.append([ctx[-2], ctx[-1]])
            owner.append(bi)
            ctx.append(int(tok))
        lens.append(len(ln))
    if not rows:
        return torch.zeros(len(lines))
    X = torch.tensor(rows, dtype=torch.long)
    logits = model.forward(X)  # (N, V)
    logp = torch.log_softmax(logits, dim=-1)
    Y = torch.tensor([int(t) for ln in lines for t in ln], dtype=torch.long)
    tok_lp = logp[torch.arange(len(Y)), Y]  # (N,)
    sums = torch.zeros(len(lines))
    for i, lp in zip(owner, tok_lp):
        sums[i] = sums[i] + lp
    return sums


def _batch_line_ce_sums(model, openings, lines):
    """Same as above but as positive CE sums (=-logprob). Differentiable."""
    return -_batch_line_logprobs(model, openings, lines)


def train_sft(start_model, pairs, cfg, seed=0):
    torch.manual_seed(seed + 7); random.seed(seed + 7)
    import copy
    model = copy.deepcopy(start_model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.sft_lr)
    # Pre-flatten winners to trigram rows once -> fast SFT = MLE on winner tokens
    Xs, Ys = [], []
    for p in pairs:
        ctx = list(p["opening"])
        for tok in p["winner"]:
            Xs.append([ctx[-2], ctx[-1]])
            Ys.append(int(tok))
            ctx.append(int(tok))
    import torch as _t
    Xs = _t.tensor(Xs, dtype=_t.long); Ys = _t.tensor(Ys, dtype=_t.long)
    n = len(Ys)
    model.train()
    for step in range(cfg.sft_steps):
        idx = torch.randint(0, n, (cfg.batch_size * 8,))  # token-batch
        opt.zero_grad()
        loss = F.cross_entropy(model(Xs[idx]), Ys[idx])
        loss.backward()
        opt.step()
    return model


def train_dpo(start_model, pairs, cfg, seed=0, log_every=100):
    torch.manual_seed(seed + 13); random.seed(seed + 13)
    import copy
    model = copy.deepcopy(start_model)
    ref = copy.deepcopy(start_model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.dpo_lr)
    hist = []
    model.train()
    B = min(cfg.batch_size, len(pairs))
    for step in range(cfg.dpo_steps):
        idx = random.sample(range(len(pairs)), B)
        ops = [pairs[i]["opening"] for i in idx]
        ws = [pairs[i]["winner"] for i in idx]
        ls = [pairs[i]["loser"] for i in idx]
        with torch.no_grad():
            ref_w = -_batch_line_ce_sums(ref, ops, ws)  # logprobs
            ref_l = -_batch_line_ce_sums(ref, ops, ls)
        # differentiable policy sums (single forward for w+l combined)
        pi_w = -_batch_line_ce_sums(model, ops, ws)
        pi_l = -_batch_line_ce_sums(model, ops, ls)
        gap = (pi_w - ref_w) - (pi_l - ref_l)
        u = cfg.beta * gap
        loss = F.softplus(-u).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0 or step == 0:
            hist.append({"step": step + 1, "loss": loss.item()})
    return model, ref, hist
