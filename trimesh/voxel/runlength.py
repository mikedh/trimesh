"""
Numpy encode/decode/utility implementations for run length encodings.

# Run Length Encoded Features

Encoding/decoding functions for run length encoded data.

We include code for two variations:

* run length encoding (RLE)
* binary run length encdoing (BRLE)

RLE stores sequences of repeated values as the value followed by its count, e.g.

```python
dense_to_rle([5, 5, 3, 2, 2, 2, 2, 6]) == [5, 2, 3, 1, 2, 4, 6, 1]
```

i.e. the value `5` is repeated `2` times, then `3` is repeated `1` time, `2` is
repeated `4` times and `6` is repeated `1` time.

BRLE is an optimized form for when the stored values can only be `0` or `1`.
This means we only need to save the counts, and assume the values alternate
(starting at `0`).

```python
dense_to_brle([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]) == \
              [0, 2, 4, 7, 2]
```

i.e. the value zero occurs `0` times, followed by `2` ones, `4` zeros, `7` ones
and `2` zeros.

Sequences with counts exceeding the data type's maximum value have to be
handled carefully. For example, the `uint8` encoding of 300 zeros
(`uint8` has a max value of 255) is:

* RLE: `[0, 255, 0, 45]`  (`0` repeated `255` times + `0` repeated `45` times)
* BRLE: `[255, 0, 45, 0]` (`255` zeros + `0` ones + `45` zeros + `0` ones)

This module contains implementations of various RLE/BRLE operations and
`RunLengthEncoding` and `BinaryRunLengthEncoding` classes.
"""
import functools
import numpy as np


def brle_length(brle):
    """Optimized implementation of `len(brle_to_dense(brle))`"""
    return np.sum(brle)


def rle_length(rle):
    """Optimized implementation of `len(rle_to_dense(rle_to_brle(rle)))`"""
    return np.sum(rle[1::2])


def rle_to_brle(rle, dtype=None):
    """Convert run length encoded (RLE) value/counts to BRLE.

    RLE data is stored in a rank 1 array with each pair giving (value, count)

    e.g. the RLE encoding of [4, 4, 4, 1, 1, 6] is [4, 3, 1, 2, 6, 1].

    Args:
        rle: run length encoded data

    Returns:
        equivalent binary run length encoding. a list if dtype is None,
            otherwise brle_to_brle is called on that list before returning.

    Raises:
        ValueError if any of the even counts of `rle` are not zero or 1.
    """
    curr_val = 0
    out = [0]
    acc = 0
    for value, count in np.reshape(rle, (-1, 2)):
        acc += count
        if value not in (0, 1):
            raise ValueError(
                "Invalid run length encoding for conversion to BRLE")
        if value == curr_val:
            out[-1] += count
        else:
            out.append(int(count))
            curr_val = value
    if len(out) % 2:
        out.append(0)
    if dtype is not None:
        out = brle_to_brle(out, dtype=dtype)
    out = maybe_pad_brle(out)
    return out


def brle_logical_not(brle):
    """Get the BRLE encoding of the `logical_not`ed dense form of `brle`.

    Equivalent to `dense_to_brle(np.logical_not(brle_to_dense(brle)))` but
    highly optimized - just pads brle with a 0 on each end (or strips is
    existing endpoints are both zero).

    Args:
        brle: rank 1 int array of
    """
    if brle[0] or brle[-1]:
        return np.pad(brle, [1, 1], mode='constant')
    else:
        return brle[1:-1]


def maybe_pad_brle(lengths, start_value=False):
    """Get a potentially padded version of lengths.

    Args:
        lengths: rank 1 int array
        start_value: bool indicating value corresponding to the first value of
            lengths

    Returns:
        rank 1 array of same dtype as lengths, with an extra zero at the front
            if `start_value`, and an extra zero at the end if the resulting array
            would not have an even number of elements.
    """
    pad_left = int(start_value)
    pad_right = (len(lengths) + pad_left) % 2
    if pad_left + pad_right > 0:
        return np.pad(lengths, [pad_left, pad_right], mode='constant')
    else:
        return lengths


def merge_brle_lengths(lengths):
    """Inverse of split_long_brle_lengths."""
    if len(lengths) == 0:
        return []

    out = [int(lengths[0])]
    accumulating = False
    for length in lengths[1:]:
        if accumulating:
            out[-1] += length
            accumulating = False
        else:
            if length == 0:
                accumulating = True
            else:
                out.append(int(length))
    return maybe_pad_brle(out)


def split_long_brle_lengths(lengths, dtype=np.int64):
    """Split lengths that exceed max dtype value.

    Lengths `l` are converted into [max_val, 0] * l // max_val + [l % max_val]

    e.g. for dtype=np.uint8 (max_value == 255)
    ```
    split_long_brle_lengths([600, 300, 2, 6], np.uint8) == \
             [255, 0, 255, 0, 90, 255, 0, 45, 2, 6]
    ```
    """
    lengths = np.asarray(lengths)
    max_val = np.iinfo(dtype).max
    bad_length_mask = lengths > max_val
    if np.any(bad_length_mask):
        # there are some bad lenghs
        nl = len(lengths)
        repeats = np.asarray(lengths) // max_val
        remainders = (lengths % max_val).astype(dtype)
        lengths = np.empty(shape=(np.sum(repeats)*2 + nl,), dtype=dtype)
        np.concatenate(
            [np.array([max_val, 0] * repeat + [remainder], dtype=dtype)
             for repeat, remainder in zip(repeats, remainders)], out=lengths)
        return lengths
    elif lengths.dtype != dtype:
        return lengths.astype(dtype)
    else:
        return lengths


def dense_to_brle(dense_data, dtype=np.int64):
    """
    Get the binary run length encoding of `dense_data`.

    Args:
        dense_data: rank 1 bool array of data to encode.
        dtype: numpy int type.

    Returns:
        Binary run length encoded rank 1 array of dtype `dtype`.
    """
    if dense_data.dtype != np.bool:
        raise ValueError("`dense_data` must be bool")
    if len(dense_data.shape) != 1:
        raise ValueError("`dense_data` must be rank 1.")
    n = len(dense_data)
    starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    lengths = split_long_brle_lengths(lengths, dtype=dtype)
    return maybe_pad_brle(lengths, dense_data[0])


_ft = np.array([False, True], dtype=np.bool)


def brle_to_dense(brle_data, vals=None):
    """Decode binary run length encoded data to dense.

    Args:
        brle_data: BRLE counts of False/True values
        vals: if not `None`, a length 2 array/list/tuple with False/True substitute
            values, e.g. brle_to_dense([2, 3, 1, 0], [7, 9]) == [7, 7, 9, 9, 9, 7]

    Returns:
        rank 1 dense data of dtype `bool if vals is None else vals.dtype`
    """
    if vals is None:
        vals = _ft
    else:
        vals = np.asarray(vals)
        if vals.shape != (2,):
            raise ValueError("vals.shape must be (2,), got %s" % (vals.shape))
    ft = np.repeat(
        _ft[np.newaxis, :], (len(brle_data)+1) // 2, axis=0).flatten()
    return np.repeat(ft[:len(brle_data)], brle_data).flatten()


def rle_to_dense(rle_data, dtype=None):
    """Get the dense decoding of the associated run length encoded data."""
    values, counts = np.split(np.reshape(rle_data, (-1, 2)), 2, axis=-1)
    if dtype is not None:
        values = values.astype(values)
    return np.repeat(np.squeeze(values, axis=-1), np.squeeze(counts, axis=-1))


def dense_to_rle(dense_data, dtype=np.int64):
    """Get run length encoding of the provided dense data."""
    n = len(dense_data)
    starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = dense_data[starts]
    values, lengths = split_long_rle_lengths(values, lengths, dtype=dtype)
    out = np.stack((values, lengths), axis=1)
    return out.flatten()


def split_long_rle_lengths(values, lengths, dtype=np.int64):
    """Split long lengths in the associated run length encoding.

    e.g.
    ```python
    split_long_rle_lengths([5, 300, 2, 12], np.uint8) == [5, 255, 5, 45, 2, 12]
    ```

    Args:
        values: values column of run length encoding, or `rle[::2]`
        lengths: counts in run length encoding, or `rle[1::2]`
        dtype: numpy data type indicating the maximum value.

    Returns:
        values, lengths associated with the appropriate splits. `lengths` will
        be of type `dtype`, while `values` will be the same as the value passed
        in.
    """
    max_length = np.iinfo(dtype).max
    lengths = np.asarray(lengths)
    repeats = lengths // max_length
    if np.any(repeats):
        repeats += 1
        remainder = lengths % max_length
        values = np.repeat(values, repeats)
        lengths = np.empty(len(repeats), dtype=dtype)
        lengths.fill(max_length)
        lengths = np.repeat(lengths, repeats)
        lengths[np.cumsum(repeats)-1] = remainder
    elif lengths.dtype != dtype:
        lengths = lengths.astype(dtype)
    return values, lengths


def merge_rle_lengths(values, lengths):
    """Inverse of split_long_rle_lengths except returns normal python lists."""
    ret_values = []
    ret_lengths = []
    curr = None
    for value, length in zip(values, lengths):
        if length == 0:
            continue
        if value == curr:
            ret_lengths[-1] += length
        else:
            curr = value
            ret_lengths.append(int(length))
            ret_values.append(value)
    return ret_values, ret_lengths


def brle_to_rle(brle, dtype=np.int64):
    if len(brle) % 2 == 1:
        brle = np.concatenate([brle, [0]])
    lengths = brle
    values = np.tile(_ft, len(brle) // 2)
    return rle_to_rle(
        np.stack((values, lengths), axis=1).flatten(), dtype=dtype)


def brle_to_brle(brle, dtype=np.int64):
    """Almost the identity function.

    Checks for possible merges and required splits.
    """
    return split_long_brle_lengths(merge_brle_lengths(brle), dtype=dtype)


def rle_to_rle(rle, dtype=np.int64):
    """Almost the identity function.

    Checks for possible merges and required splits.
    """
    values, lengths = np.reshape(rle, (-1, 2)).T
    values, lengths = merge_rle_lengths(values, lengths)
    values, lengths = split_long_rle_lengths(values, lengths, dtype=dtype)
    return np.stack((values, lengths), axis=1).flatten()


def _unsorted_gatherer(indices, sorted_gather_fn):
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices, copy=False)
    order = np.argsort(indices)
    ordered_indices = indices[order]

    def f(data, dtype=None):
        ans = np.empty(len(order), dtype=dtype or getattr(data, 'dtype', None))
        ans[order] = tuple(sorted_gather_fn(data, ordered_indices))
        return ans

    return f


def sorted_rle_gather_1d(rle_data, ordered_indices):
    """
    Gather brle_data at ordered_indices.

    This is equivalent to `rle_to_dense(brle_data)[ordered_indices]` but avoids
    the decoding.

    Args:
        brle_data: iterable of run-length-encoded data.
        ordered_indices: iterable of ints in ascending order.

    Returns:
        `brle_data` iterable of values at the dense indices, same length as
        ordered indices.
    """
    data_iter = iter(rle_data)
    index_iter = iter(ordered_indices)
    index = next(index_iter)
    start = 0
    while True:
        while start <= index:
            try:
                value = next(data_iter)
                start += next(data_iter)
            except StopIteration:
                raise IndexError(
                    'Index %d out of range of raw_values length %d'
                    % (index, start))
        try:
            while index < start:
                yield value
                index = next(index_iter)
        except StopIteration:
            break


def rle_mask(rle_data, mask):
    data_iter = iter(rle_data)
    mask_iter = iter(mask)
    while True:
        try:
            value = next(data_iter)
            count = next(data_iter)
        except StopIteration:
            break
        for c in range(count):
            m = next(mask_iter)
            if m:
                yield value


def brle_mask(rle_data, mask):
    data_iter = iter(rle_data)
    mask_iter = iter(mask)
    value = True
    while True:
        try:
            value = not value
            count = next(data_iter)
        except StopIteration:
            break
        for c in range(count):
            m = next(mask_iter)
            if m:
                yield value


def rle_gatherer_1d(indices):
    """
    Get a gather function at the given indices.

    Because gathering on RLE data requires sorting, for instances where
    gathering at the same indices on different RLE data this can save the
    sorting process.

    If only gathering on a single RLE iterable, use `rle_gather_1d`.

    Args:
        indices: iterable of integers

    Returns:
        gather function, mapping `(rle_data, dtype=None) -> values`.
        `values` will have the same length as `indices` and dtype provided,
        or rle_data.dtype if no dtype is provided.
    """
    return _unsorted_gatherer(indices, sorted_rle_gather_1d)


def rle_gather_1d(rle_data, indices, dtype=None):
    """
    Gather RLE data values at the provided dense indices.

    This is equivalent to `rle_to_dense(rle_data)[indices]` but the
    implementation does not require the construction of the dense array.

    If indices is known to be in order, use `sorted_gather_1d`.

    Args:
        rle_data: run length encoded data
        indices: dense indices
        dtype: numpy dtype. If not provided, uses rle_data.dtype

    Returns:
        numpy array, dense data at indices, same length as indices and dtype as
        rle_data
    """
    return rle_gatherer_1d(indices)(rle_data, dtype=dtype)


def sorted_brle_gather_1d(brle_data, ordered_indices):
    """
    Gather brle_data at ordered_indices.

    This is equivalent to `brle_to_dense(brle_data)[ordered_indices]` but
    avoids the decoding.

    Args:
        raw_data: iterable of run-length-encoded data.
        ordered_indices: iterable of ints in ascending order.

    Returns:
        `raw_data` iterable of values at the dense indices, same length as
        ordered indices.
    """
    data_iter = iter(brle_data)
    index_iter = iter(ordered_indices)
    index = next(index_iter)
    start = 0
    value = True
    while True:
        while start <= index:
            try:
                value = not value
                start += next(data_iter)
            except StopIteration:
                raise IndexError(
                    'Index %d out of range of raw_values length %d'
                    % (index, start))
        try:
            while index < start:
                yield value
                index = next(index_iter)
        except StopIteration:
            break


def brle_gatherer_1d(indices):
    """
    Get a gather function at the given indices.

    Because gathering on BRLE data requires sorting, for instances where
    gathering at the same indices on different RLE data this can save the
    sorting process.

    If only gathering on a single RLE iterable, use `brle_gather_1d`.

    Args:
        indices: iterable of integers

    Returns:
        gather function, mapping `(rle_data, dtype=None) -> values`.
        `values` will have the same length as `indices` and dtype provided,
        or rle_data.dtype if no dtype is provided.
    """
    return functools.partial(
        _unsorted_gatherer(indices, sorted_brle_gather_1d), dtype=np.bool)


def brle_gather_1d(brle_data, indices, dtype=None):
    """
    Gather BRLE data values at the provided dense indices.

    This is equivalent to `rle_to_dense(rle_data)[indices]` but the
    implementation does not require the construction of the dense array.

    If indices is known to be in order, use `sorted_brle_gather_1d`.

    Args:
        rle_data: run length encoded data
        indices: dense indices
        dtype: numpy dtype. If not provided, uses rle_data.dtype

    Returns:
        numpy array, dense data at indices, same length as indices and dtype as
        rle_data
    """
    return brle_gatherer_1d(indices)(brle_data, dtype=dtype)


def brle_reverse(brle_data):
    """Equivalent to dense_to_brle(brle_to_dense(brle_data)[-1::-1])."""
    if len(brle_data) % 2 == 0:
        brle_data = np.concatenate([brle_data, [0]], axis=0)
    end = -1 if brle_data[-1] == 0 else None
    return brle_data[-1:end:-1]


def rle_reverse(rle_data):
    """Get the rle encoding of the reversed dense array."""
    if not isinstance(rle_data, np.ndarray):
        rle_data = np.array(rle_data, copy=False)
    rle_data = np.reshape(rle_data, (-1, 2))
    rle_data = rle_data[-1::-1]
    return np.reshape(rle_data, (-1,))


def rle_to_sparse(rle_data):
    """Get dense indices associated with non-zeros."""
    indices = []
    values = []
    it = iter(rle_data)
    index = 0
    try:
        while True:
            value = next(it)
            counts = next(it)
            end = index + counts
            if value:
                indices.append(np.arange(index, end, dtype=np.int64))
                values.append(np.repeat(value, counts))
            index = end
    except StopIteration:
        pass
    indices = np.concatenate(indices)
    values = np.concatenate(values)
    return indices, values


def brle_to_sparse(brle_data, dtype=np.int64):
    ends = np.cumsum(brle_data)
    indices = [np.arange(s, e, dtype=dtype) for s, e in
               zip(ends[::2], ends[1::2])]
    return np.concatenate(indices)


class RunLengthEncoding(object):
    def __init__(self, data):
        data = np.asanyarray(data, dtype=getattr(data, 'dtype', np.int64))
        if len(data.shape) != 1 or not isinstance(data, np.ndarray):
            raise ValueError('data must be 1D numpy array')
        self._data = data

    def sum(self):
        return (self._data[::2] * self._data[1::2]).sum()

    @staticmethod
    def from_dense(dense_data, dtype=np.int64):
        return RunLengthEncoding(dense_to_rle(dense_data, dtype=dtype))

    @staticmethod
    def from_rle(rle_data, dtype=None):
        if dtype != rle_data.dtype:
            rle_data = rle_to_rle(rle_data, dtype=dtype)
        return RunLengthEncoding(rle_data)

    @staticmethod
    def from_brle(brle_data, dtype=None):
        return RunLengthEncoding(brle_to_rle(brle_data, dtype=dtype))

    def dense_length(self):
        return rle_length(self._data)

    def reverse(self):
        return rle_reverse(self._data)

    def sparse_indices(self):
        return rle_to_sparse(self._data)[0]

    def to_dense(self):
        return rle_to_dense(self._data)

    def gather(self, indices, dtype=None):
        return rle_gather_1d(self._data, indices, dtype=dtype)

    def sorted_gather(self, ordered_indices, dtype=None):
        return np.array(
            tuple(sorted_rle_gather_1d(self._data, ordered_indices)),
            dtype=dtype)

    def mask(self, mask, dtype=None):
        return np.array(tuple(rle_mask(self._data, mask)), dtype=dtype)

    def get_value(self, index):
        for value in self.sorted_gather((index,)):
            return value


class BinaryRunLengthEncoding(RunLengthEncoding):
    @staticmethod
    def from_dense(dense_data, dtype=np.int64):
        return BinaryRunLengthEncoding(dense_to_brle(dense_data, dtype))

    @staticmethod
    def from_rle(rle_data, dtype=None):
        return BinaryRunLengthEncoding(rle_to_brle(rle_data, dtype=dtype))

    @staticmethod
    def from_brle(brle_data, dtype=None):
        if dtype != brle_data.dtype:
            brle_data = brle_to_brle(brle_data, dtype=dtype)
        return BinaryRunLengthEncoding(brle_data)

    def sum(self):
        return self._data[1::2].sum()

    def dense_length(self):
        return brle_length(self._data)

    def reverse(self):
        return brle_reverse(self._data)

    def sparse_indices(self):
        return brle_to_sparse(self._data)

    def to_dense(self):
        return brle_to_dense(self._data)

    def gather(self, indices, dtype=np.bool):
        return brle_gather_1d(self._data, indices, dtype=dtype)

    def sorted_gather(self, ordered_indices, dtype=np.bool):
        return np.array(
            tuple(sorted_brle_gather_1d(self._data, ordered_indices)),
            dtype=dtype)

    def mask(self, mask, dtype=np.bool):
        return np.array(tuple(brle_mask(self._data, mask)), dtype=dtype)
