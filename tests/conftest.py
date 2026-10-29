import numpy as np
import pytest


def global_rng_state() -> tuple:
    """
    Capture the global numpy RNG state as comparable values.

    Returns
    ------------
    state : tuple
      The MT19937 keys, position, and cached gaussian.
    """
    name, keys, position, has_gauss, cached = np.random.get_state()
    return name, keys.tobytes(), position, has_gauss, cached


@pytest.fixture(autouse=True)
def no_global_rng():
    """
    Fail any test which draws from or reseeds the global numpy RNG.

    The global stream is process-wide so a draw in one test shifts the
    values every later test sees. That coupling used to wedge CI: a seed
    leaked out of `test_registration`, `test_cap_nohit` rode the stream
    it left behind, and a rotation from it produced a polygon that hung
    `mapbox_earcut` in native code until the job timed out.

    Tests that want random-looking data should use `generic.RandomSeed`
    or an explicit `np.random.default_rng(seed)`. Note `generic.random`
    is a fixed array rather than a stream: it's fine for one-shot bulk
    data but two calls in a scope return the same values.
    """
    before = global_rng_state()
    yield
    if global_rng_state() != before:
        raise AssertionError(
            "test touched the global numpy RNG: use `generic.RandomSeed`, `generic.random_transforms`, or `np.random.default_rng(seed)`"
        )
