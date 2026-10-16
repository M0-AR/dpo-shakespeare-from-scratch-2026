import math
from dpo_lab.loss import dpo_loss_single, dpo_gap, loss_for_done_threshold

def test_exercise_pair():
    before = dpo_loss_single(-10.3, -10.3, -22.8, -22.8, beta=0.1)
    assert abs(before - 0.6931) < 1e-3, before
    after = dpo_loss_single(-15.0, -10.3, -64.3, -22.8, beta=0.1)
    assert abs(after - 0.025) < 0.005, after
    gap = dpo_gap(-15.0, -10.3, -64.3, -22.8)
    assert abs(gap - 36.8) < 1e-9, gap

def test_done_threshold():
    u = loss_for_done_threshold(0.1)
    assert abs(u - 2.25) < 0.02, u
    # beta*gap = u -> gap = u/beta
    assert abs(u / 0.1 - 22.5) < 0.2
    assert abs(u / 0.01 - 225) < 2
    assert abs(u / 0.5 - 4.5) < 0.05
