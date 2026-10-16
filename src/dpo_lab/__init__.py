"""DPO Shakespeare lab — tiny word-level LM + rule judge + SFT vs DPO."""
from .model import TinyWordLM
from .judge import Judge, HAPPY_WORDS, SAD_WORDS
from .loss import dpo_loss_single, dpo_gap, sigmoid

__all__ = ["TinyWordLM", "Judge", "HAPPY_WORDS", "SAD_WORDS", "dpo_loss_single", "dpo_gap", "sigmoid"]
