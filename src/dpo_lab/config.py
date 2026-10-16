"""Central hyper-parameters mirroring the video."""
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 0
    vocab_size: int = 4000          # words known (excl. specials)
    embed_dim: int = 64
    hidden_dim: int = 256
    context_len: int = 2            # opening = 2 words ("I am", "content to", ...)
    max_line_len: int = 12          # words after opening (excl. EOS)
    lm_epochs: int = 6
    lm_batch: int = 512
    lm_lr: float = 3e-3
    n_pairs_target: int = 78000     # sampled openings x2 lines (video: ~78k)
    train_pairs_keep: int = 10593   # approx after tie-filter (video exact)
    sft_steps: int = 600
    dpo_steps: int = 600
    batch_size: int = 64
    sft_lr: float = 3e-3
    dpo_lr: float = 1e-3
    beta: float = 0.1
    eval_openings: int = 1000
    data_url: str = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    # Complete Works of Shakespeare, Project Gutenberg ebook #100 (public domain)

DEFAULT = Config()
