from ragcheck.dataset import anchor_score, classify_difficulty, leakage_score

ANSWER = "The billing service retries failed charges three times before opening a ticket."


def test_quoting_query_leaks_everything() -> None:
    assert leakage_score(ANSWER, ANSWER) == 1.0


def test_short_question_leaks_little() -> None:
    assert leakage_score("What happens when a charge fails?", ANSWER) < 0.4


def test_anchor_score_reflects_verbatim_overlap() -> None:
    assert anchor_score("billing service retries charges", ANSWER) == 1.0
    assert anchor_score("payment problems escalation policy", ANSWER) == 0.0


def test_degenerate_inputs_count_as_fully_leaked() -> None:
    assert leakage_score("what is this", "the of and") == 1.0
    assert anchor_score("the of and", ANSWER) == 1.0


def test_difficulty_tiers() -> None:
    assert classify_difficulty(0.9) == "easy"
    assert classify_difficulty(0.75) == "easy"
    assert classify_difficulty(0.5) == "medium"
    assert classify_difficulty(0.2) == "hard"
