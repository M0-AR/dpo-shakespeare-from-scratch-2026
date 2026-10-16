from dpo_lab.judge import Judge

def test_judge_examples():
    j = Judge()
    # video examples: "do truth, for I had myself" scores 0
    assert j.score("do truth for i had myself".split()) == 0
    # "just die" scores -1
    assert j.score("just die".split()) == -1
    # "glad to see it so" scores +1
    assert j.score("glad to see it so".split()) == 1
    w, sa, sb = j.pick("do truth for i had myself".split(), "just die".split())
    assert w == 0

def test_lexicon_size():
    from dpo_lab.judge import HAPPY_WORDS, SAD_WORDS
    assert len(HAPPY_WORDS) == 18 and len(SAD_WORDS) == 18
