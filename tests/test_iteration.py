from collections import OrderedDict
from functools import reduce

import numpy as np

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
        assert [current.index(k) for k in keys] == list(range(len(keys)))

    # exercise every path which can insert a key
    d = IndexedDict({"a": 1, "b": 2})
    d["c"] = 3
    d.update({"d": 4}, e=5)
    d.setdefault("f", 6)
    d |= {"g": 7}
    check(d)
    assert d["f"] == 6

    # setting an existing key must not move it or grow the dict
    d["a"] = 10
    check(d)
    assert d.index("a") == 0
    # callers which used to be handed an `OrderedDict` must not break
    assert isinstance(d, OrderedDict)
    # and a copy must not degrade into a plain `dict`
    assert isinstance(d.copy(), IndexedDict)
    check(d.copy())

    # removing or reordering would shift every position after it so it
    # must raise loudly rather than silently returning stale indexes
    for name in ("__delitem__", "pop", "popitem", "move_to_end"):
        try:
            getattr(d, name)("a")
            raise AssertionError(f"`{name}` should have raised!")
        except TypeError:
            pass
    # the failed removals must not have altered anything
    check(d)

    # `clear` and `update` is the supported way to remove
    d.clear()
    assert len(d) == 0
    d.update({"z": 1, "y": 2})
    check(d)
    assert d.index("y") == 1


if __name__ == "__main__":
    test_reduce_cascade()
