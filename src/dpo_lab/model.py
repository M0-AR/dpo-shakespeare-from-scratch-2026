"""Tiny word-level LM: P(next | prev-2 words). CPU-fast, from scratch."""
import torch
import torch.nn as nn


class TinyWordLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 256, context_len: int = 2):
        super().__init__()
        self.vocab_size = vocab_size  # includes specials
        self.context_len = context_len
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(context_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: (B, C) -> logits (B, V)
        e = self.emb(ctx).flatten(1)
        return self.net(e)

    @torch.no_grad()
    def logprob_of_line(self, opening_ids, line_ids) -> float:
        """Sum log P(line + EOS | opening). line_ids excludes opening, includes EOS at end."""
        self.eval()
        ctx = list(opening_ids)
        total = 0.0
        for tok in line_ids:
            x = torch.tensor([ctx[-self.context_len:]], dtype=torch.long)
            logits = self.forward(x)[0]
            lp = torch.log_softmax(logits, dim=-1)[tok].item()
            total += lp
            ctx.append(tok)
        return total
