"""End-to-end: data -> LM -> pairs -> SFT -> DPO -> eval -> RESULTS.json + figures."""
import json, random, sys, math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dpo_lab.config import Config
from dpo_lab.data import build_corpus
from dpo_lab.train import train_lm, train_sft, train_dpo
from dpo_lab.pairs import make_pairs, sample_line, ids_to_words
from dpo_lab.evaluate import happy_share, win_rate, surprise_on_heldout, vocab_used, implicit_reward_accuracy
from dpo_lab.loss import dpo_loss_single

def next_token_probs(model, ctx_ids):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor([ctx_ids]))[0], dim=-1)

def main(out="results", seed=0, fast=False):
    cfg = Config(seed=seed)
    if fast:
        cfg.lm_epochs = 2; cfg.sft_steps = 200; cfg.dpo_steps = 200
        cfg.n_pairs_target = 12000
    outp = ROOT / out; outp.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); random.seed(seed)

    print("[1/7] live public data (Gutenberg #100) ...", flush=True)
    corpus = build_corpus(cfg.data_url, ROOT / "results" / "pg100.txt",
                          n_lines=22000, vocab_keep=cfg.vocab_size, seed=seed)
    print(f"  source={corpus['source']} sha={corpus['sha']} V={corpus['V']}", flush=True)

    print("[2/7] train start-model LM ...", flush=True)
    start = train_lm(corpus, cfg, seed=seed)
    torch.save(start.state_dict(), outp / "start.pt")

    print("[3/7] sample pairs + judge ...", flush=True)
    # video: ~78k openings -> ties thrown away -> 10,593 kept
    bundle = make_pairs(start, corpus, n_openings=cfg.n_pairs_target, seed=seed + 1)
    pairs = bundle["pairs"]
    print(f"  openings={bundle['n_openings']} kept={len(pairs)} ties={bundle['ties']}", flush=True)
    # deterministic subsample to ~10.5k to mirror video scale
    if len(pairs) > 10593:
        rnd = random.Random(seed + 2)
        rnd.shuffle(pairs); pairs = pairs[:10593]
    # split: train pairs + 233 held-out pairs from unseen openings for reward test
    held_pairs = pairs[-233:] if len(pairs) >= 500 else pairs
    train_pairs = pairs[:-233] if len(pairs) >= 500 else pairs

    print("[4/7] SFT on winners (copy-the-winners) ...", flush=True)
    sft = train_sft(start, train_pairs, cfg, seed=seed)
    torch.save(sft.state_dict(), outp / "sft.pt")

    print("[5/7] DPO ...", flush=True)
    dpo, ref, hist = train_dpo(start, train_pairs, cfg, seed=seed)
    torch.save(dpo.state_dict(), outp / "dpo.pt")
    print(f"  dpo hist: {hist}", flush=True)

    print("[6/7] eval on 1000 fresh openings ...", flush=True)
    rnd = random.Random(seed + 5)
    cands = [l for l in corpus["encoded"] if len(l) >= 4]
    rnd.shuffle(cands)
    eval_ops = [l[:2] for l in cands[:cfg.eval_openings]]

    hs_start = happy_share(start, eval_ops, corpus)
    hs_sft = happy_share(sft, eval_ops, corpus)
    hs_dpo = happy_share(dpo, eval_ops, corpus)
    wr_sft = win_rate(sft, start, eval_ops, corpus)
    wr_dpo = win_rate(dpo, start, eval_ops, corpus)
    sur_start = surprise_on_heldout(start, corpus)
    sur_sft = surprise_on_heldout(sft, corpus)
    sur_dpo = surprise_on_heldout(dpo, corpus)
    vocab_dpo, dpo_lines = vocab_used(dpo, eval_ops, corpus)
    vocab_start, _ = vocab_used(start, eval_ops, corpus)
    acc = implicit_reward_accuracy(dpo, ref, held_pairs)

    # video's worked pair check: recompute exercise numbers
    ex_before = dpo_loss_single(-10.3, -10.3, -22.8, -22.8, beta=0.1)
    ex_after = dpo_loss_single(-15.0, -10.3, -64.3, -22.8, beta=0.1)

    # targeted micro-probe: P(glad | 'I am') and P(END | context) before/after
    w2i = corpus["w2i"]; i2w = corpus["i2w"]
    def prob_word(model_, ctx_words, target):
        try:
            ctx = [w2i.get(w, corpus["UNK"]) for w in ctx_words]
            tid = w2i.get(target, corpus["UNK"])
            p = next_token_probs(model_, ctx)[tid].item()
            return p
        except Exception:
            return float("nan")
    # find 'i am' ids if present else skip
    glad_start = prob_word(start, ["i", "am"], "glad")
    glad_dpo = prob_word(dpo, ["i", "am"], "glad")
    eos_start = next_token_probs(start, eval_ops[0]).detach().numpy()[corpus["EOS"]]
    eos_dpo = next_token_probs(dpo, eval_ops[0]).detach().numpy()[corpus["EOS"]]

    # actual log-probs of one real held-out pair under start/ref/dpo
    hp = held_pairs[0]
    def lp(m, op, ids):
        ctx = list(op); t = 0.0
        import torch as _t
        with _t.no_grad():
            for tok in ids:
                x = _t.tensor([ctx[-2:]], dtype=_t.long)
                t += _t.log_softmax(m.forward(x)[0], dim=-1)[tok].item()
                ctx.append(tok)
        return t
    real_w_start = lp(start, hp["opening"], hp["winner"])
    real_l_start = lp(start, hp["opening"], hp["loser"])
    real_w_dpo = lp(dpo, hp["opening"], hp["winner"])
    real_l_dpo = lp(dpo, hp["opening"], hp["loser"])

    results = {
        "config": cfg.__dict__,
        "data_source": corpus["source"], "data_sha": corpus["sha"],
        "n_openings_sampled": bundle["n_openings"], "n_pairs_kept": len(pairs),
        "n_ties": bundle["ties"],
        "happy_share": {"start": hs_start, "sft": hs_sft, "dpo": hs_dpo},
        "win_rate_vs_start": {"sft": wr_sft, "dpo": wr_dpo},
        "surprise_heldout": {"start": sur_start, "sft": sur_sft, "dpo": sur_dpo},
        "vocab_used_1000": {"start": vocab_start, "dpo": vocab_dpo},
        "implicit_reward_acc_233": acc,
        "dpo_loss_hist": hist,
        "exercise_check": {"loss_before_0.693": ex_before, "loss_after_0.025": ex_after,
                           "gap_36.8_beta_gap_3.68": 36.8},
        "micro_probe": {"P_glad_given_I_am_start": glad_start, "P_glad_given_I_am_dpo": glad_dpo,
                        "P_EOS_start": float(eos_start), "P_EOS_dpo": float(eos_dpo)},
        "real_pair_logprobs": {"w_start": real_w_start, "l_start": real_l_start,
                               "w_dpo": real_w_dpo, "l_dpo": real_l_dpo},
        "dpo_sample_lines": dpo_lines[:12],
        "video_targets": {
            "happy": [0.05, 0.28, 0.98], "win": [0.29, 0.965],
            "surprise": [190, 218, 1768], "vocab": [1029, 770],
            "loss_before": 0.693, "loss_after": 0.025, "reward_acc": 0.99,
        },
    }
    (outp / "RESULTS.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[7/7] wrote {outp/'RESULTS.json'}", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    main(out=a.out, seed=a.seed, fast=a.fast)
