import numpy as np

from recommender import mmr


def _identical_pair():
    """Candidates 0 and 1 are duplicates; 2 is unrelated and slightly worse."""
    scores = np.array([0.90, 0.89, 0.60])
    pairwise = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return scores, pairwise


def test_zero_diversity_is_plain_top_k():
    scores, pairwise = _identical_pair()
    assert list(mmr.select(scores, pairwise, k=3, diversity=0.0)) == [0, 1, 2]


def test_diversity_breaks_up_near_duplicates():
    scores, pairwise = _identical_pair()
    assert list(mmr.select(scores, pairwise, k=2, diversity=0.5)) == [0, 2]


def test_the_best_candidate_is_always_first():
    scores, pairwise = _identical_pair()
    for diversity in (0.0, 0.3, 0.7, 1.0):
        assert mmr.select(scores, pairwise, k=3, diversity=diversity)[0] == 0


def test_selection_has_no_repeats_and_respects_k():
    rng = np.random.default_rng(0)
    scores = rng.random(30)
    block = rng.random((30, 30))
    picked = mmr.select(scores, (block + block.T) / 2, k=10, diversity=0.4)

    assert len(picked) == len(set(picked.tolist())) == 10


def test_k_larger_than_the_pool_is_clamped():
    scores, pairwise = _identical_pair()
    assert len(mmr.select(scores, pairwise, k=99, diversity=0.3)) == 3
    assert len(mmr.select(scores, pairwise, k=0, diversity=0.3)) == 0
