from collections import OrderedDict
from functools import reduce

import numpy as np
import pytest

from trimesh.iteration import IndexedDict, chain, reduce_cascade


def test_reduce_cascade():
    # the multiply will explode quickly past the integer maximum
    def both(operation, items):
        """
        Run our cascaded reduce and regular reduce.
        """

        b = reduce_cascade(operation, items)

        if len(items) > 0:
            assert b == reduce(operation, items)

        return b

    for i in range(20):
        data = np.arange(i)
        c = both(items=data, operation=lambda a, b: a + b)

        if i == 0:
            assert c is None
        else:
            assert c == np.arange(i).sum()

        # try a multiply
        data = np.arange(i)
        c = both(items=data, operation=lambda a, b: a * b)

        if i == 0:
            assert c is None
        else:
            assert c == np.prod(data)

        # try a multiply
        data = np.arange(i)[1:]
        c = both(items=data, operation=lambda a, b: a * b)
        if i <= 1:
            assert c is None
        else:
            assert c == np.prod(data)

    data = ["a", "b", "c", "d", "e", "f", "g"]
    print("# reduce_pairwise\n-----------")
    r = both(operation=lambda a, b: a + b, items=data)

    assert r == "abcdefg"


def test_chain():
    # should work on iterables the same as `itertools.chain`
    assert np.allclose(chain([1, 3], [4]), [1, 3, 4])
    # should work with non-iterable single values
    assert np.allclose(chain([1, 3], 4), [1, 3, 4])
    # should filter out `None` arguments
    assert np.allclose(chain([1, 3], None, 4, None), [1, 3, 4])


def test_indexed_dict():
    def check(current):
        # `index` must agree with the slow version it exists to replace
        # for every key, which is the only thing it promises
        keys = list(current.keys())
        assert [current.index(key) for key in keys] == list(range(len(keys)))

    # exercise every path which can insert a key
    indexed = IndexedDict({"a": 1, "b": 2})
    indexed["c"] = 3
    indexed.update({"d": 4}, e=5)
    indexed.setdefault("f", 6)
    indexed |= {"g": 7}
    # setting an existing key must not move it or grow the dict
    indexed["a"] = 10
    check(indexed)
    assert len(indexed) == 7
    assert indexed["f"] == 6 and indexed["a"] == 10 and indexed.index("a") == 0

    # callers handed an `OrderedDict` must not break and a copy must not
    # degrade into a plain `dict` which would forget every position
    assert isinstance(indexed, OrderedDict) and isinstance(indexed.copy(), IndexedDict)
    check(indexed.copy())

    # removing or reordering would shift every position after it so it
    # must raise loudly rather than silently returning stale indexes
    for name in ("__delitem__", "pop", "popitem", "move_to_end"):
        with pytest.raises(TypeError):
            getattr(indexed, name)("a")
    # the failed removals must not have altered anything
    check(indexed)

    # `clear` and `update` is the supported way to remove
    indexed.clear()
    indexed.update({"z": 1, "y": 2})
    check(indexed)
    assert len(indexed) == 2 and indexed.index("y") == 1


if __name__ == "__main__":
    test_reduce_cascade()
