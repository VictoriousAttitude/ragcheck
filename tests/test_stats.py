import pytest

from ragcheck.stats import paired_bootstrap_ci


def test_point_is_the_mean_difference() -> None:
    ci = paired_bootstrap_ci([0.0, -0.2, -0.4, 0.0])
    assert ci.point == pytest.approx(-0.15)


def test_identical_systems_have_a_degenerate_interval() -> None:
    ci = paired_bootstrap_ci([0.0, 0.0, 0.0, 0.0])
    assert (ci.point, ci.low, ci.high) == (0.0, 0.0, 0.0)


def test_constant_difference_has_zero_width_interval() -> None:
    ci = paired_bootstrap_ci([-0.1, -0.1, -0.1, -0.1, -0.1])
    assert ci.low == pytest.approx(-0.1)
    assert ci.high == pytest.approx(-0.1)


def test_interval_brackets_the_point() -> None:
    ci = paired_bootstrap_ci([0.5, -0.5, 0.3, -0.7, 0.1], resamples=2000, seed=1)
    assert ci.low <= ci.point <= ci.high
    assert ci.low < ci.high


def test_seed_makes_the_interval_reproducible() -> None:
    deltas = [0.2, -0.3, 0.0, -0.5, 0.4, -0.1]
    assert paired_bootstrap_ci(deltas, seed=7) == paired_bootstrap_ci(deltas, seed=7)


def test_single_observation_collapses_to_the_point() -> None:
    ci = paired_bootstrap_ci([-0.3])
    assert (ci.point, ci.low, ci.high) == (-0.3, -0.3, -0.3)


def test_rejects_empty_and_bad_confidence() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, 0.2], confidence=1.5)
