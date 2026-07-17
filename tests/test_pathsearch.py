import numpy as np

from mcp_solver.path.pathsearch import pathsearch


def _bp(ts, xs):
    return [(t, np.array([x])) for t, x in zip(ts, xs)]


def test_accepts_endpoint_when_good():
    # merit = |x|; endpoint x=0 is perfect
    bps = _bp([0.0, 0.5, 1.0], [4.0, 2.0, 0.0])
    got = pathsearch(lambda x: abs(float(x[0])), bps, reference=4.0, sigma=0.01)
    assert got is not None
    x, t, m = got
    assert t == 1.0 and m == 0.0


def test_walks_back_when_endpoint_bad():
    # merit spikes at the endpoint; the middle breakpoint is acceptable
    def merit(x):
        return abs(float(x[0]))
    bps = _bp([0.0, 0.6, 1.0], [4.0, 1.0, 50.0])
    got = pathsearch(merit, bps, reference=4.0, sigma=0.01)
    assert got is not None
    x, t, m = got
    assert m <= (1 - 0.01 * t) * 4.0
    assert m <= 1.0 + 1e-12          # found the good middle region


def test_segment_armijo_finds_interior_point():
    # endpoint region undefined (inf merit); an interior point of the last
    # segment must be located by halving from the far end
    def merit(x):
        v = float(x[0])
        if v > 3.5:
            return np.inf            # undefined region near the endpoint
        return abs(v - 2.0) + 1.0    # best merit 1.0 at v = 2
    bps = _bp([0.0, 1.0], [0.0, 4.0])
    got = pathsearch(merit, bps, reference=2.0, sigma=0.01)
    assert got is not None
    x, t, m = got
    assert np.isfinite(m) and m <= (1 - 0.01 * min(max(t, 0), 1)) * 2.0
    assert float(x[0]) <= 3.5


def test_returns_none_when_nothing_acceptable():
    bps = _bp([0.0, 0.5, 1.0], [5.0, 6.0, 7.0])
    got = pathsearch(lambda x: abs(float(x[0])), bps, reference=1.0, sigma=0.01)
    assert got is None


def test_negative_t_clamped_in_acceptance_factor():
    # t < 0 must not INFLATE the acceptance threshold above reference
    bps = _bp([0.0, -0.5], [4.0, 3.999])
    got = pathsearch(lambda x: abs(float(x[0])), bps, reference=4.0, sigma=0.5)
    assert got is not None            # 3.999 <= (1 - 0.5*0) * 4.0
    x, t, m = got
    assert m <= 4.0


def _merit_capped(x):
    # merit = |x|, but anything past 100 reads as an undefined (inf) region
    v = float(x[0])
    if v > 100.0:
        return np.inf
    return abs(v)


def test_zero_t_point_with_equal_merit_rejected():
    # A later breakpoint numerically coincident with the search's start
    # (t == 0.0, merit == reference exactly) must NOT be accepted as
    # progress -- that would be a zero-length / null step. With the bad
    # endpoint's merit undefined (inf), and every interior point on either
    # side strictly worse than reference, pathsearch must return None.
    bps = _bp([0.0, 0.0, 1.0], [4.0, 4.0, 1000.0])
    got = pathsearch(_merit_capped, bps, reference=4.0, sigma=0.01)
    assert got is None


def test_zero_t_point_with_strict_improvement_accepted():
    # Same shape as above, but the t == 0.0 breakpoint's merit is strictly
    # below the reference -- this IS real progress and must be accepted.
    bps = _bp([0.0, 0.0, 1.0], [4.0, 3.9, 1000.0])
    got = pathsearch(_merit_capped, bps, reference=4.0, sigma=0.01)
    assert got is not None
    x, t, m = got
    assert t == 0.0 and m == 3.9
