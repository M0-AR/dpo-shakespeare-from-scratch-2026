"""DPO loss for one pair — the video's exercise.

Given four log-probs:
  logpi_w, logref_w, logpi_l, logref_l
rise_w = logpi_w - logref_w
rise_l = logpi_l - logref_l
gap    = rise_w - rise_l
loss   = -log(sigmoid(beta * gap))

Video's worked pair (opening "I am"):
  before: winner -10.3, loser -22.8 (ref == policy, rises 0, gap 0, loss 0.693)
  after 600 DPO steps: winner -15.0, loser -64.3
  rise_w = -4.7, rise_l = -41.5, gap = 36.8, beta*gap = 3.68, loss ~= 0.025
"""
import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def dpo_gap(logpi_w: float, logref_w: float, logpi_l: float, logref_l: float) -> float:
    return (logpi_w - logref_w) - (logpi_l - logref_l)


def dpo_loss_single(logpi_w: float, logref_w: float,
                    logpi_l: float, logref_l: float,
                    beta: float = 0.1) -> float:
    gap = dpo_gap(logpi_w, logref_w, logpi_l, logref_l)
    return -math.log(sigmoid(beta * gap) + 1e-30)


def loss_for_done_threshold(loss_target: float = 0.1) -> float:
    """beta*gap needed so that loss == target. For 0.1 -> ~2.25."""
    # loss = -log(sigmoid(u)) = softplus(-u); solve softplus(-u)=0.1
    # u = -log(exp(0.1)-1)
    return -math.log(math.exp(loss_target) - 1.0)
