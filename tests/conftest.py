import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_global_rng():
    """
    Fail any test which draws from or reseeds the global numpy RNG, as a
    seed leaked between tests once hung `mapbox_earcut` and wedged CI: use
    `generic.RandomSeed` or `np.random.default_rng(seed)` instead.
    """
    before = np.random.get_state(legacy=False)
    yield
    after = np.random.get_state(legacy=False)
    assert repr(after) == repr(before), "test touched the global numpy RNG"
