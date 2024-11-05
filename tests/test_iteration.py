from functools import reduce

import numpy as np

from trimesh.iteration import chain, reduce_cascade


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


if __name__ == "__main__":
    test_reduce_cascade()
