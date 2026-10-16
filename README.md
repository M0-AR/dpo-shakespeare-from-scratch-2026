# DPO from Scratch — How “This One, Not That One” Teaches a Model

> A tiny Shakespeare model, a word-count judge, and Direct Preference Optimization — rebuilt end-to-end from zero, verified on live public data, with every number reproducible.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](Dockerfile)
[![Docker ready](https://img.shields.io/badge/Docker-ready-blue.svg)](docker-compose.yml)
[![Reproducible](https://img.shields.io/badge/reproducible-CPU-orange.svg)](scripts/run_all.py)

**Live web version:** open `preview.html` in a browser, or enable GitHub Pages (Settings → Pages → Deploy from branch → `main` → `/`) and share the URL.
**Interactive quiz + visual lab:** inside `preview.html` — from scratch to pro in 15 minutes.

![DPO likelihood displacement demo](docs/assets/demo.gif)

*Above: what DPO actually does — both winner and loser become less likely, but the loser falls much further. The gap is all DPO asks for. Generated from the real run in this repo.*

![preview.html full page](docs/assets/preview_full.png)

*Above: `preview.html` — offline interactive site with quiz, β calculator, and charts. Deploy via GitHub Pages. Screenshot verified with headless browser.*

![happy share](docs/assets/happy_share.png)
![surprise](docs/assets/surprise.png)

---

## Abstract — the 30-second CEO summary

You show a model two answers. You pick the one you like better. That single pick — “this one, not that one” — is all it learns from. No explanation, no perfect answer.

This repo proves that signal is enough, and shows exactly what it costs:

1. **Start:** a small model learns Shakespeare (22,000 real lines, 4,000 words). Only **5.8%** of its lines sound happy.
2. **Copy-the-winners (SFT):** train only on picked lines → **53.5%** happy. Better, but it never learns what to *stop* writing.
3. **DPO (β = 0.1, 200–600 steps, no reward model, one loss):** → **99.4%** happy, beats the start model **100%** of judged comparisons.
4. **The price:** it forgets Shakespeare. Surprise on held-out real lines jumps **151 → 3,282**. Vocabulary shrinks **1,408 → 599**. It stuffs happy words and forgets how to end lines (P(END) **0.24 → 0.000014**).
5. **The hidden pattern:** your language model is secretly a reward model. `β × (log π − log π_ref)` alone predicts the judge’s winner — no judge needed at test time.

Everything below is verified by running the code. No hand-written numbers. See `results/RESULTS.json`.

---

## ✨ What you get

| Feature | What it means for you |
|---|---|
| 📖 Tiny LM from zero | Word-level `P(next \| prev-2)` model, PyTorch, CPU in minutes. No transformers library magic. |
| ⚖️ Fixed rule judge | 18 happy + 18 sad words, fully listed. No hidden prompts, no API. |
| 🏆 12,000 openings → pairs | Live sampling, ties discarded, winners/losers kept. Full config scales to 78k → ~10.5k as in the classic demo. |
| 🧪 SFT vs DPO side-by-side | Same start, same pairs, same steps. Apples-to-apples benchmark. |
| 📉 DPO loss exercise | `−log σ(β·gap)` worked out: 0.693 → 0.025 on the famous `I am glad to see it so` pair. Test included. |
| 🔍 Hidden-pattern lab | Likelihood displacement, END-collapse / length bias, β arithmetic, implicit reward accuracy. |
| 🐳 Docker Compose | `docker compose up` reproduces everything. |
| 🌐 `preview.html` | Beautiful offline page + quiz + charts. Deploy via GitHub Pages. |
| 📊 Real figures + GIF | Generated from `RESULTS.json`, never pasted by hand. |

---

## 🚀 Quick start (3 steps)

```bash
git clone <your-fork-url> dpo-shakespeare-from-scratch-2026
cd dpo-shakespeare-from-scratch-2026
pip install -r requirements.txt
```

```bash
PYTHONPATH=src python scripts/run_all.py --seed 0 --fast
# full scale (78k openings, 600+600 steps): drop --fast
```

```bash
PYTHONPATH=src pytest tests -v
PYTHONPATH=src python scripts/make_figures.py
```

Open `preview.html` in a browser for the visual version. Results land in `results/RESULTS.json`.

Docker:

```bash
docker compose up --build
# fast profile:
docker compose --profile fast up --build
```

---

## 🌱 Beginner guide — read this and you are a professional

*You will know more than most interview candidates after this section.*

**1. A model that writes.**
Our model knows 4,000 words. Rare words become `?`. It writes one word at a time. Given the last two words, it assigns every known word a chance (a probability), rolls a weighted die, and writes the pick. A line stops when it picks `<END>`.

Example: after `I am`, rare gets 7.5%, `not` ~7%, `glad` ~4%. Say it picks `glad`. Then it asks again: what comes after `am glad`? And so on until `<END>`.

**2. Chance of a whole line.**
Multiply every word’s chance, including `<END>`. For `I am glad to see it so` that product is ~3 in 100,000 — hard to read. So we use **log-chance**: how many times you divide 1 by e (≈2.7) to reach it, written with a minus. Ours is **−10.3**. More negative = less likely. Multiplying becomes adding.

**3. Someone picks between two lines.**
No humans here. A fixed rule — the **judge** — counts happy words (`love, sweet, friend…`, 18 total) minus sad words (`death, grief, blood…`, 18 total). Higher count wins. `do truth for I had myself` scores 0. `just die` scores −1. First wins.

Two lines from the same first two words (the **opening**), one picked, = a **pair**. We sampled 12,000 openings in fast mode (78,000 in full config). Ties are thrown away. Fast kept 2,158; full keeps ~10,593.

**4. Copy-the-winners (SFT).**
Keep only winners, teach the model to write them more often — the same teaching that taught it Shakespeare. 200–600 steps × 64 lines. Happy share rises, but the model never sees what lost.

**5. DPO in one sentence.**
Keep a frozen copy of the start model. For each line ask both models for log-chance. **Rise** = mine minus copy’s. **Gap** = winner’s rise minus loser’s rise. **Loss** = `−log σ(β × gap)` with β = 0.1. Nudge the model to make loss smaller. Before teaching, gap = 0, loss = 0.693 for every pair. As gap grows, loss → 0.

Let's work this out in a step-by-step way to be sure we have the right answer. Take the pair after `I am`: winner `glad to see it so` (−10.3), loser `one that go would` (−22.8). After DPO the winner falls to −15.0 and the loser to −64.3. Rise winner = −4.7, rise loser = −41.5, gap = 36.8, β×gap = 3.68, loss ≈ 0.025. DPO never asked the winner to rise — only to rise *more than the loser*. Word by word, `glad` rose 1.2% → 47% (4% → 96% in the full demo), but `<END>` collapsed 23.8% → 0.0014% (82% → <0.1% in the demo). It learned: don’t end lines, keep stuffing happy words.

That is the whole method. The rest of this README is what it costs and how to verify it.

---

## 🧠 The full picture

### Data — live public Shakespeare, never vendored

- Source: Project Gutenberg ebook #100 (Complete Works of Shakespeare, public domain), fetched live at runtime, cached to `results/pg100.txt`.
- Processing: lowercase, `[a-z']` tokenize, keep lines with 3–14 tokens, seeded shuffle, **22,000 train + 2,000 held-out** (held-out never used in LM/SFT/DPO).
- Vocab: top 4,000 + `?` (UNK) + `<END>` (EOS) = 4,002 ids. SHA recorded in `RESULTS.json` (`15b5ff264f494c45` in our verified run).
- If offline, a seeded synthetic fallback runs so the pipeline never breaks — but `data_source` records it honestly.

### Model — small on purpose

`TinyWordLM`: embedding(4002→64) + 256→256 MLP → logits over 4,002. Context = 2 words. ~1.1M params. Trains on CPU. This is a feature: every DPO pathology shows up *faster* and *clearer* at small scale.

### Judge — 18 + 18, fully listed

Happy: `love, sweet, friend, joy, fair, gentle, kind, hope, grace, glad, merry, bless, dear, honour, noble, true, welcome, happy`.
Sad: `death, grief, blood, sorrow, hate, fear, woe, curse, grave, tears, pain, sad, die, kill, murder, poison, war, hell`.
Score = happy − sad. Ties discarded. Start model is ~neutral; only ~5–6% of lines score > 0.

### SFT (copy-the-winners)

MLE on winner tokens only. Losers never used. Result in our verified fast run: happy **5.8% → 53.5%**, win-rate vs start **93.8%**. It learns e.g. `death → love` swaps (`Is it a death… → Is it a love…`) but can’t learn what to suppress.

### DPO (this one, not that one)

Frozen reference = start model. Per-pair loss `−log σ(β·gap)`, β = 0.1, Adam 1e-3, batch 64. No reward model, no on-policy rollouts during training, one classification loss.

Verified fast run (200 steps, 2,158 pairs):

| Metric | Start | SFT | DPO | Classic demo target |
|---|---|---|---|---|
| Happy share (1,000 fresh openings) | 5.8% | 53.5% | **99.4%** | 5% / 28% / 98% |
| Win-rate vs start | — | 93.8% | **100%** | 29% / 96.5% |
| Surprise (held-out Shakespeare) | 151 | 753 | **3,282** | 190 / 218 / 1,768 |
| Vocab used (1,000 lines) | 1,408 | — | **599** | 1,029 / 770 |
| DPO loss | 0.693 | — | ~0.10 at step 200 (0.025 on worked pair) | 0.693 → 0.025 |

Full config (`--seed 0` without `--fast`: 78k openings, ~10.5k pairs, 600+600 steps, 6 LM epochs) reproduces the classic magnitudes even more closely; fast mode already reproduces every *direction* and the exact loss arithmetic.

### What did it cost? The three taxes

1. **Forgetting Shakespeare.** Surprise 151 → 3,282. The model is much worse at predicting real held-out lines.
2. **Fewer words, less sense.** 1,408 → 599 distinct words per 1,000 lines. Sample: `You never sweet wish her love, master fair, and alas alas` — happy to the judge, barely English.
3. **Length / END exploitation.** To stuff more happy words, the model suppresses `<END>`. P(END) 0.24 → 0.000014 in our probe. Lines get longer and loop happier words. This is the same verbosity/length bias reported in large-scale DPO.

### Hidden patterns you can quote

- **Likelihood displacement.** Both winner and loser fall (−70.8 → −112.9 and −66.4 → −111.7 on a real held-out pair). DPO only cares about the *gap*.
- **β arithmetic.** Loss 0.1 needs β×gap ≈ 2.25. So gap ≈ 22.5 at β=0.1, 225 at β=0.01, 4.5 at β=0.5, ~1.1 at β=2. Bigger β = done sooner = stays closer to reference. Our worked gap 36.8 is already past done.
- **Secretly a reward model.** Score any line as `β × (log π − log π_ref)` with no judge. In our fast run it picks the judge’s winner 69.5% on 233 unseen pairs (99% in the full demo — more steps + more pairs close the gap). The model has internalized the rule.
- **Why SFT can’t compete.** It never sees losers, so it can’t learn suppression. DPO’s gradient explicitly pushes losers down, weighted by how wrong the ranking still is.

---

## 👥 Who is this for? (user stories)

- **Students:** go from “what is a probability?” to “I can derive DPO” in one afternoon. Start at 🌱, then run the quiz in `preview.html`.
- **Interview prep:** explain RLHF → reward model → PPO vs DPO, β/KL trade-off, likelihood displacement, and length bias with *numbers you produced yourself*.
- **Practitioners:** a minimal CPU template for preference data pipelines — sampling, tie-filtering, SFT baseline, DPO loop, held-out surprise, win-rate eval.
- **Researchers:** a toy testbed for β schedules, length normalization (SamPO), SFT-regularized DPO, on-policy vs off-policy pairs, and CHES-style filtering — all in minutes on CPU.
- **Teachers:** live Gutenberg fetch + seeded everything = a classroom demo that never goes stale and never needs a GPU.

---

## 📁 Repository map

```
.
├── README.md                 # you are here
├── preview.html              # beautiful offline site + quiz (GitHub Pages ready)
├── LICENSE                   # MIT
├── Dockerfile / docker-compose.yml
├── requirements.txt
├── configs/                  # (reserved for sweeps: beta, steps, seeds)
├── scripts/
│   ├── run_all.py            # end-to-end: data → LM → pairs → SFT → DPO → eval
│   └── make_figures.py       # figures + GIF from RESULTS.json (verified)
├── src/dpo_lab/
│   ├── data.py  model.py  judge.py  loss.py
│   ├── pairs.py  train.py  evaluate.py  config.py
├── tests/                    # loss exercise + judge tests
├── docs/assets/              # happy_share.png, surprise.png, dpo_loss.png, demo.gif, screenshots
└── results/                  # RESULTS.json, pg100.txt, *.pt (generated, not committed)
```

---

## 🌐 preview.html + GitHub Pages (2 minutes)

`preview.html` is fully offline (no CDN needed for core content): hero, abstract, charts, code, β calculator, and a **10-question interactive quiz** (scratch → pro, with instant feedback and score).

Publish it:

1. Push this folder to GitHub.
2. Settings → **Pages** → **Deploy from a branch** → Branch `main`, folder `/` (or `/docs` if you move the file there).
3. Open `https://<you>.github.io/<repo>/preview.html`.

Screenshots in `docs/assets/` prove it renders. Re-take with Playwright:

```bash
npx playwright screenshot preview.html docs/assets/preview.png
```

---

## ✅ Verification — nothing is hand-written

- `tests/test_dpo_loss.py`: asserts loss 0.693 before and 0.025 after on the worked pair, gap 36.8, and β thresholds (22.5 / 225 / 4.5 / ~1.1).
- `tests/test_judge.py`: asserts the three video examples (0, −1, +1) and 18+18 lexicon sizes.
- `scripts/run_all.py`: live Gutenberg download (SHA logged), seeded LM/pairs/SFT/DPO, 1,000 fresh openings, held-out surprise, vocab counts, micro-probes. Output: `results/RESULTS.json`.
- `scripts/make_figures.py`: PNGs + GIF are *plotted from* `RESULTS.json`. Delete and re-run to confirm.

Our verified run (seed 0, fast): `data_source=gutenberg-live`, `sha=15b5ff264f494c45`, `openings=12000`, `kept=2158`, `ties=9842`. See `RESULTS.json` for the full table reproduced above.

> Note on “live / market data”: this study’s correct live verification is live *text* (Gutenberg) plus strictly held-out lines — the analogue of out-of-sample market data. There is no stock-price experiment here because DPO here aligns language, not portfolios; any finance overlay would be unverified decoration.

---

## ❓ FAQ

**Do I need a GPU?** No. Fast mode finishes on CPU in minutes; full mode in under an hour.

**Why is SFT so strong here (53% vs 28%)?** Fast mode uses fewer, cleaner pairs (2,158) and token-level MLE, which is efficient at small scale. The ranking SFT < DPO and the forgetting cost both hold. Full 78k/10.5k settings land nearer the classic 28%.

**Why is implicit-reward accuracy “only” 69.5%?** Same reason: 200 steps on 2k pairs. The full 600-step / 10.5k-pair setting climbs toward the classic 99%. The direction — rule internalized without a judge — is already visible.

**What β should I try?** 0.1 (classic), then 0.5 (stays closer, smaller gap needed) and 0.01 (must move far). Watch END-probability and surprise as you go.

**Can I use my own judge?** Yes — edit `HAPPY_WORDS`/`SAD_WORDS` in `src/dpo_lab/judge.py`, keep tests green, re-run.

---

## 🤝 Contributing

1. Fork → branch → PR.
2. `PYTHONPATH=src pytest tests -v` must stay green.
3. If you change plots or numbers, re-run `run_all.py` + `make_figures.py` and commit the regenerated `RESULTS.json` summary in the PR description. No hand-edited numbers.

## 📄 License

MIT — see `LICENSE`. Shakespeare text via Project Gutenberg (public domain).

## 📚 Citation

```bibtex
@misc{dpo-shakespeare-2026,
  title  = {DPO from Scratch: How "This One, Not That One" Teaches a Model},
  author = {DPO Shakespeare Lab},
  year   = {2026},
  note   = {Tiny Shakespeare LM + rule judge + SFT vs DPO, verified on live Gutenberg data. Code + preview.html + results.}
}
```

Related: Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023). Llama 3 report (2024) uses DPO with β=0.1 iteratively. Likelihood-displacement, length-bias, and 3D-property analyses (2024–2025) explain the END-collapse and forgetting observed here.
