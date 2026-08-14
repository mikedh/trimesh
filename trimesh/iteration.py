from collections import OrderedDict
from math import log2
from typing import Any

from .typed import Callable, Iterable, NDArray, Sequence


def reduce_cascade(operation: Callable, items: Sequence | NDArray):
    """
    Call an operation function in a cascaded pairwise way against a
    flat list of items.

    This should produce the same result as `functools.reduce`
    if `operation` is commutable like addition or multiplication.
    This may be faster for an `operation` that runs with a speed
    proportional to its largest input, which mesh booleans appear to.

    The union of a large number of small meshes appears to be
    "much faster" using this method.

    This only differs from `functools.reduce` for commutative `operation`
    in that it returns `None` on empty inputs rather than `functools.reduce`
    which raises a `TypeError`.

    For example on `a b c d e f g` this function would run and return:
        a b
        c d
        e f
        ab cd
        ef g
        abcd efg
     -> abcdefg

    Where `functools.reduce` would run and return:
        a b
        ab c
        abc d
        abcd e
        abcde f
        abcdef g
     -> abcdefg

    Parameters
    ----------
    operation
      The function to call on pairs of items.
    items
      The flat list of items to apply operation against.
    """
    if len(items) == 0:
        return None
    elif len(items) == 1:
        # skip the loop overhead for a single item
        return items[0]
    elif len(items) == 2:
        # skip the loop overhead for a single pair
        return operation(items[0], items[1])

    for _ in range(int(1 + log2(len(items)))):
        results = []

        # loop over pairs of items.
        items_mod = len(items) % 2
        for i in range(0, len(items) - items_mod, 2):
            results.append(operation(items[i], items[i + 1]))

        # if we had a non-even number of items it will have been
        # skipped by the loop so append it to our list
        if items_mod != 0:
            results.append(items[-1])

        items = results

    # logic should have reduced to a single item
    assert len(results) == 1

    return results[0]


def chain(*args: Iterable[Any] | Any | None) -> list[Any]:
    """
    A less principled version of `list(itertools.chain(*args))` that
    accepts non-iterable values, filters `None`, and returns a list
    rather than yielding values.

    If all passed values are iterables this will return identical
    results to `list(itertools.chain(*args))`.


    Examples
    ----------

    In [1]: list(itertools.chain([1,2], [3]))
    Out[1]: [1, 2, 3]

    In [2]: trimesh.util.chain([1,2], [3])
    Out[2]: [1, 2, 3]

    In [3]: trimesh.util.chain([1,2], [3], 4)
    Out[3]: [1, 2, 3, 4]

    In [4]: list(itertools.chain([1,2], [3], 4))
      ----> 1 list(itertools.chain([1,2], [3], 4))
      TypeError: 'int' object is not iterable

    In [5]: trimesh.util.chain([1,2], None, 3, None, [4], [], [], 5, [])
    Out[5]: [1, 2, 3, 4, 5]


    Parameters
    -----------
    args
      Will be individually checked to see if they're iterable
      before either being appended or extended to a flat list.


    Returns
    ----------
    chained
      The values in a flat list.
    """
    # collect values to a flat list
    chained = []
    # extend if it's a sequence, otherwise append
    [
        chained.extend(a)
        if (hasattr(a, "__iter__") and not isinstance(a, (str, bytes)))
        else chained.append(a)
        for a in args
        if a is not None
    ]
    return chained


class IndexedDict(OrderedDict):
    """
    An append-only `OrderedDict` which knows what position a key was inserted at.

    Useful anywhere values are referenced by *position* but keyed by content so
    duplicates are only stored once: the only other spelling is
    `list(d.keys()).index(key)`, which allocates every key and scans it, i.e.
    quadratic. Looking up the position of all `n` keys once each:

        n       list(keys()).index()      this class
        2000          0.061s                0.00007s
        4000          0.262s                0.00012s
        8000          1.135s                0.00026s

    Removing or reordering a key would shift the position of every key after it,
    so `__delitem__`, `pop`, `popitem`, and `move_to_end` raise: the supported
    way to remove is `clear` followed by `update`.

    Examples
    ----------

    In [1]: IndexedDict({"a": 1, "b": 2, "c": 3}).index("c")
    Out[1]: 2
    """

    # subclasses `OrderedDict` rather than `dict` as it routes `__init__`, `update`,
    # `setdefault`, `|=`, and `copy` through `__setitem__`: one place records a position

    def __init__(self, *args, **kwargs):
        # must exist before `super` starts routing through `__setitem__`
        self._position = {}
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key not in self:
            self._position[key] = len(self)
        super().__setitem__(key, value)

    def _forbidden(self, *args, **kwargs):
        raise TypeError("`IndexedDict` is append-only: use `clear` and `update`")

    # removing or reordering a key shifts the position of every key after it
    __delitem__ = pop = popitem = move_to_end = _forbidden

    def clear(self) -> None:
        super().clear()
        self._position.clear()

    def index(self, key) -> int:
        """
        Which position in insertion order was `key` inserted at.
        """
        return self._position[key]
