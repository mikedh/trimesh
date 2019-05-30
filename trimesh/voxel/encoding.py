"""OO interfaces to encodings for ND arrays wich caching."""
import numpy as np
import abc
import sys
from . import runlength as rl
from .. import caching
# TODO Sparse Encoding
# Options:
# - scipy.sparse.coo_matrix - how to generalize to nd?
# - https://pypi.org/project/sparse/0.1.1/ -
# - separate indices / values - see commented out implementation below
# - {index: value} dict
#   - efficient DataStore implementation?
#   - deterministic index ordering?

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Encoding(ABC):
    def __init__(self, data):
        self._data = data
        self._cache = caching.Cache(id_function=data.crc)

    @abc.abstractproperty
    def dtype(self):
        pass

    @abc.abstractproperty
    def shape(self):
        pass

    @abc.abstractproperty
    def sum(self):
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractproperty
    def sparse_indices(self):
        pass

    @abc.abstractproperty
    def sparse_values(self):
        pass

    @abc.abstractproperty
    def dense(self):
        pass

    @abc.abstractmethod
    def gather_nd(self, indices):
        pass

    @abc.abstractmethod
    def mask(self, mask):
        pass

    @abc.abstractmethod
    def get_value(self, index):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    @caching.cache_decorator
    def stripped(self):
        """
        Get encoding with all zeros stripped from the start/end of each axis.

        Returns:
            encoding:
            padding: (n, 2) array of ints denoting padding at the start/end
                that was stripped
        """
        dense = self.dense
        shape = dense.shape
        ndims = len(shape)
        padding = []
        slices = []
        for dim, size in enumerate(shape):
            axis = tuple(range(dim)) + tuple(range(dim + 1, ndims))
            filled = np.any(dense, axis=axis)
            indices, = np.nonzero(filled)
            pad_left = indices[0]
            pad_right = indices[-1]
            padding.append([pad_left, pad_right])
            slices.append(slice(pad_left, pad_right))
        return DenseEncoding(dense[tuple(slices)]), np.array(padding, int)

    def _flip(self, axes):
        return FlippedEncoding(self, axes)

    def md5(self):
        return self._data.md5()

    def crc(self):
        return self._data.crc()

    @property
    def ndims(self):
        return len(self.shape)

    def reshape(self, shape):
        return self.flat if len(shape) == 1 else ShapedEncoding(self, shape)

    @property
    def flat(self):
        return FlattenedEncoding(self)

    def flip(self, axis=0):
        return _flipped(self, axis)

    @property
    def sparse_components(self):
        return self.sparse_indices, self.sparse_values

    @property
    def data(self):
        return self._data

    def run_length_data(self, dtype=np.int64):
        if self.ndims != 1:
            raise ValueError(
                '`run_length_data` only valid for flat encodings')
        return rl.dense_to_rle(self.dense, dtype=dtype)

    def binary_run_length_data(self, dtype=np.int64):
        if self.ndims != 1:
            raise ValueError(
                '`run_length_data` only valid for flat encodings')
        return rl.dense_to_brle(self.dense, dtype=dtype)

    def transpose(self, perm):
        return _transposed(self, perm)

    def _transpose(self, perm):
        return TransposedEncoding(self, perm)


class DenseEncoding(Encoding):
    """Simple `Encoding` implementation based on a numpy ndarray."""

    def __init__(self, data):
        if not isinstance(data, caching.TrackedArray):
            if not isinstance(data, np.ndarray):
                raise ValueError(
                    'DenseEncoding data should be a numpy array, got object of'
                    ' type %s' % type(data))
            data = caching.tracked_array(data)
        super().__init__(data=data)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @caching.cache_decorator
    def sum(self):
        return self._data.sum()

    @property
    def size(self):
        return self._data.size

    @property
    def sparse_components(self):
        indices = self.sparse_indices
        values = self.gather(indices)
        return indices, values

    @caching.cache_decorator
    def sparse_indices(self):
        return np.column_stack(np.where(self._data))

    @caching.cache_decorator
    def sparse_values(self):
        return self.sparse_components[1]

    def _flip(self, axes):
        dense = self.dense
        for a in axes:
            dense = np.flip(dense, a)
        return DenseEncoding(dense)

    @property
    def dense(self):
        return self._data

    def gather(self, indices):
        return self._data[indices]

    def gather_nd(self, indices):
        return self._data[tuple(indices.T)]

    def mask(self, mask):
        return self._data[mask if isinstance(mask, np.ndarray) else mask.dense]

    def get_value(self, index):
        return self._data[tuple(index)]

    def reshape(self, shape):
        return DenseEncoding(self._data.reshape(shape))

    def _transpose(self, perm):
        return DenseEncoding(self._data.transpose(perm))

    @property
    def flat(self):
        return DenseEncoding(self._data.reshape((-1,)))

    def copy(self):
        return DenseEncoding(self._data.copy())


# class SparseEncoding(Encoding):
#     def __init__(self, indices, values, shape):
#         data = caching.DataStore()
#         super().__init__(data)
#         data['indices'] = indices
#         data['values'] = values
#         self._shape = shape

#     @property
#     def sparse_indices(self):
#         return self._data['indices']

#     @property
#     def sparse_values(self):
#         return self._data['values']

#     @property
#     def dtype(self):
#         return self.sparse_values.dtype

#     @caching.cache_decorator
#     def sum(self):
#         return self.sparse_values.sum()

#     @property
#     def ndims(self):
#         return self.sparse_indices.shape[-1]

#     def shape(self):
#         return self._shape

#     @property
#     def size(self):
#         return self.sparse_values.size

#     @property
#     def sparse_components(self):
#         return self.sparse_indices, self.sparse_values

#     @caching.cache_decorator
#     def dense(self):
#         out = np.zeros(self.shape, dtype=self.dtype)
#         for i, v in zip(*self.sparse_components):
#             out[tuple(i)] = v
#         return out

#     def gather(self, indices):
#         raise NotImplementedError('TODO')

#     def mask(self, mask):
#         if isinstance(mask, np.ndarray):
#             mask = DenseEncoding(mask)
#         mask_indices = set(mask.sparse_indices)
#         indices, values = self.sparse_components
#         mask = [tuple(i) in mask_indices for i in indices]
#         return SparseEncoding(indices[mask], values[mask], self.shape)

#     def get_value(self, index):
#         for i, value_index in enumerate(self.sparse_indices):
#             if np.all(index == value_index):
#                 if index.size > 1:
#                     index = tuple(index)
#                 return self.sparse_values[index]
#         return np.zeros((), dtype=self.dtype)


class RunLengthEncoding(Encoding):
    """1D run length encoding.

    See `trimesh.voxel.runlength` documentation for implementation details.
    """

    def __init__(self, data, dtype=None):
        super().__init__(data=caching.tracked_array(data))
        if dtype is None:
            dtype = self._data.dtype
        if len(self._data.shape) != 1:
            raise ValueError('data must be 1D numpy array')
        self._dtype = dtype

    @property
    def ndims(self):
        return 1

    @property
    def shape(self):
        return (self.size,)

    @property
    def dtype(self):
        return self._dtype

    def md5(self):
        return self._data.md5()

    def crc(self):
        return self._data.crc()

    @staticmethod
    def from_dense(dense_data, dtype=np.int64, encoding_dtype=np.int64):
        return RunLengthEncoding(
            rl.dense_to_rle(dense_data, dtype=encoding_dtype), dtype=dtype)

    @staticmethod
    def from_rle(rle_data, dtype=None):
        if dtype != rle_data.dtype:
            rle_data = rl.rle_to_rle(rle_data, dtype=dtype)
        return RunLengthEncoding(rle_data)

    @staticmethod
    def from_brle(brle_data, dtype=None):
        return RunLengthEncoding(rl.brle_to_rle(brle_data, dtype=dtype))

    @caching.cache_decorator
    def stripped(self):
        data, padding = rl.rle_strip(self._data)
        if padding == (0, 0):
            encoding = self
        else:
            encoding = RunLengthEncoding(data, dtype=self._dtype)
        padding = np.expand_dims(padding, axis=0)
        return encoding, padding

    @caching.cache_decorator
    def sum(self):
        return (self._data[::2] * self._data[1::2]).sum()

    @caching.cache_decorator
    def size(self):
        return rl.rle_length(self._data)

    def _flip(self, axes):
        if axes != (0,):
            raise ValueError(
                'encoding is 1D - cannot flip on axis %s' % str(axes))
        return RunLengthEncoding(rl.rle_reverse(self._data))

    @caching.cache_decorator
    def sparse_components(self):
        return rl.rle_to_sparse(self._data)

    @caching.cache_decorator
    def sparse_indices(self):
        return self.sparse_components[0]

    @caching.cache_decorator
    def sparse_values(self):
        return self.sparse_components[1]

    @caching.cache_decorator
    def dense(self):
        return rl.rle_to_dense(self._data, dtype=self._dtype)

    def gather(self, indices):
        return rl.rle_gather_1d(self._data, indices, dtype=self._dtype)

    def gather_nd(self, indices):
        indices = np.squeeze(indices, axis=-1)
        return self.gather(indices)

    def sorted_gather(self, ordered_indices):
        return np.array(
            tuple(rl.sorted_rle_gather_1d(self._data, ordered_indices)),
            dtype=self._dtype)

    def mask(self, mask):
        return np.array(
            tuple(rl.rle_mask(self._data, mask)), dtype=self._dtype)

    def get_value(self, index):
        for value in self.sorted_gather((index,)):
            return np.asanyarray(value, dtype=self._dtype)

    def copy(self):
        return RunLengthEncoding(self._data.copy())

    def run_length_data(self, dtype=np.int64):
        return rl.rle_to_rle(self._data, dtype=dtype)

    def binary_run_length_data(self, dtype=np.int64):
        return rl.rle_to_brle(self._data, dtype=dtype)


class BinaryRunLengthEncoding(RunLengthEncoding):
    """1D binary run length encoding.

    See `trimesh.voxel.runlength` documentation for implementation details.
    """

    def __init__(self, data):
        super().__init__(data=data, dtype=bool)

    @staticmethod
    def from_dense(dense_data, encoding_dtype=np.int64):
        return BinaryRunLengthEncoding(
            rl.dense_to_brle(dense_data, dtype=encoding_dtype))

    @staticmethod
    def from_rle(rle_data, dtype=None):
        return BinaryRunLengthEncoding(
            rl.rle_to_brle(rle_data, dtype=dtype))

    @staticmethod
    def from_brle(brle_data, dtype=None):
        if dtype != brle_data.dtype:
            brle_data = rl.brle_to_brle(brle_data, dtype=dtype)
        return BinaryRunLengthEncoding(brle_data)

    @caching.cache_decorator
    def stripped(self):
        data, padding = rl.rle_strip(self._data)
        if padding == (0, 0):
            encoding = self
        else:
            encoding = BinaryRunLengthEncoding(data)
        padding = np.expand_dims(padding, axis=0)
        return encoding, padding

    @caching.cache_decorator
    def sum(self):
        return self._data[1::2].sum()

    @caching.cache_decorator
    def size(self):
        return rl.brle_length(self._data)

    def _flip(self, axes):
        if axes != (0,):
            raise ValueError(
                'encoding is 1D - cannot flip on axis %s' % str(axes))
        return BinaryRunLengthEncoding(rl.brle_reverse(self._data))

    @property
    def sparse_components(self):
        return self.sparse_indices, self.sparse_values

    @caching.cache_decorator
    def sparse_values(self):
        return np.ones(shape=(self.sum,), dtype=bool)

    @caching.cache_decorator
    def sparse_indices(self):
        return rl.brle_to_sparse(self._data)

    @caching.cache_decorator
    def dense(self):
        return rl.brle_to_dense(self._data)

    def gather(self, indices):
        return rl.brle_gather_1d(self._data, indices, dtype=bool)

    def gather_nd(self, indices):
        indices = np.squeeze(indices)
        return self.gather(indices)

    def sorted_gather(self, ordered_indices):
        gen = rl.sorted_brle_gather_1d(self._data, ordered_indices)
        return np.array(tuple(gen), dtype=bool)

    def mask(self, mask):
        gen = rl.brle_mask(self._data, mask)
        return np.array(tuple(gen), dtype=bool)

    def copy(self):
        return BinaryRunLengthEncoding(self._data.copy())

    def run_length_data(self, dtype=np.int64):
        return rl.brle_to_rle(self._data, dtype=dtype)

    def binary_run_length_data(self, dtype=np.int64):
        return rl.brle_to_brle(self._data, dtype=dtype)


class LazyIndexMap(Encoding):
    """
    Abstract class for implementing lazy index mapping operations.

    Implementations include transpose, flatten/reshaping and flipping

    Derived classes must implement:
        * _to_base_indices(indices)
        * _from_base_indices(base_indices)
        * shape
        * dense
        * mask(mask)
    """

    @abc.abstractmethod
    def _to_base_indices(self, indices):
        pass

    @abc.abstractmethod
    def _from_base_indices(self, base_indices):
        pass

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def sum(self):
        return self._data.sum

    @property
    def size(self):
        return self._data.size

    @property
    def sparse_indices(self):
        return self._from_base_indices(self._data.sparse_indices)

    @property
    def sparse_values(self):
        return self._data.sparse_values

    def gather_nd(self, indices):
        return self._data.gather_nd(self._to_base_indices(indices))

    def get_value(self, index):
        return self._data[tuple(self._to_base_indices(index))]


class FlattenedEncoding(LazyIndexMap):
    """
    Lazily flattened encoding.

    Dense equivalent is np.reshape(data, (-1,)) (np.flatten creates a copy).
    """

    def _to_base_indices(self, indices):
        return np.column_stack(np.unravel_index(indices, self._data.shape))

    def _from_base_indices(self, base_indices):
        return np.expand_dims(
            np.ravel_multi_index(base_indices.T, self._data.shape), axis=-1)

    @property
    def shape(self):
        return self.size,

    @property
    def dense(self):
        return self._data.dense.reshape((-1,))

    def mask(self, mask):
        return self._data.mask(mask.reshape(self._data.shape))

    @property
    def flat(self):
        return self

    def copy(self):
        return FlattenedEncoding(self._data.copy())


class ShapedEncoding(LazyIndexMap):
    """
    Lazily reshaped encoding.

    Numpy equivalent is `np.reshape`
    """

    def __init__(self, encoding, shape):
        if isinstance(encoding, Encoding):
            if encoding.ndims != 1:
                encoding = encoding.flat
        else:
            raise ValueError('encoding must be an Encoding')
        super().__init__(data=encoding)
        self._shape = tuple(shape)
        nn = self._shape.count(-1)
        size = np.prod(self._shape)
        if nn == 1:
            size = np.abs(size)
            if self._data.size % size != 0:
                raise ValueError(
                    'cannot reshape encoding of size %d into shape %s' %
                    (self._data.size, str(self._shape)))
            rem = self._data.size // size
            self._shape = tuple(rem if s == -1 else s for s in self._shape)
        elif nn > 2:
            raise ValueError('shape cannot have more than one -1 value')
        elif np.prod(self._shape) != self._data.size:
            raise ValueError(
                'cannot reshape encoding of size %d into shape %s' %
                (self._data.size, str(self._shape)))

    def _from_base_indices(self, base_indices):
        return np.column_stack(np.unravel_index(base_indices, self.shape))

    def _to_base_indices(self, indices):
        return np.expand_dims(
            np.ravel_multi_index(indices.T, self.shape), axis=-1)

    @property
    def flat(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def dense(self):
        return self._data.dense.reshape(self.shape)

    def mask(self, mask):
        return self._data.mask(mask.flat)

    def copy(self):
        return ShapedEncoding(encoding=self._data.copy(), shape=self.shape)


class TransposedEncoding(LazyIndexMap):
    """
    Lazily transposed encoding

    Dense equivalent is `np.transpose`
    """

    def __init__(self, base_encoding, perm):
        if not isinstance(base_encoding, Encoding):
            raise ValueError(
                'base_encoding must be an Encoding, got %s'
                % str(base_encoding))
        if len(base_encoding.shape) != len(perm):
            raise ValueError(
                'base_encoding has %d ndims - cannot transpose with perm %s'
                % (base_encoding.ndims, str(perm)))
        super().__init__(base_encoding)
        perm = np.array(perm, dtype=np.int64)
        if not all(i in perm for i in range(base_encoding.ndims)):
            raise ValueError('perm %s is not a valid permutation' % str(perm))
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(base_encoding.ndims)
        self._perm = perm
        self._inv_perm = inv_perm

    def transpose(self, perm):
        return _transposed(self._data, [self._perm[p] for p in perm])

    def _transpose(self, perm):
        raise RuntimeError('Should not be here')

    @property
    def perm(self):
        return self._perm

    @property
    def shape(self):
        shape = self._data.shape
        return tuple(shape[p] for p in self._perm)

    def _to_base_indices(self, indices):
        return np.take(indices, self._perm, axis=-1)

    def _from_base_indices(self, base_indices):
        return np.take(base_indices, self._inv_perm, axis=-1)

    @property
    def dense(self):
        return self._data.dense.transpose(self._perm)

    def gather(self, indices):
        return self._data.gather(self._base_indices(indices))

    def mask(self, mask):
        return self._data.mask(
            mask.transpose(self._inv_perm)).transpose(self._perm)

    def get_value(self, index):
        return self._data[tuple(self._base_indices(index))]

    @property
    def data(self):
        return self._data

    def copy(self):
        return TransposedEncoding(
            base_encoding=self._data.copy(), perm=self._perm)


class FlippedEncoding(LazyIndexMap):
    """
    Encoding with entries flipped along one or more axes.

    Dense equivalent is `np.flip`
    """

    def __init__(self, encoding, axes):
        ndims = encoding.ndims
        if isinstance(axes, np.ndarray) and axes.size == 1:
            axes = axes.item(),
        elif isinstance(axes, int):
            axes = axes,
        axes = tuple(a + ndims if a < 0 else a for a in axes)
        self._axes = tuple(sorted(axes))
        if len(set(self._axes)) != len(self._axes):
            raise ValueError(
                "Axes cannot contain duplicates, got %s" % str(self._axes))
        super().__init__(encoding)
        if not all(0 <= a < self._data.ndims for a in axes):
            raise ValueError(
                'Invalid axes %s for %d-d encoding'
                % (str(axes), self._data.ndims))

    def _to_base_indices(self, indices):
        indices = indices.copy()
        shape = self.shape
        for a in self._axes:
            indices[:, a] *= -1
            indices[:, a] += shape
        return indices

    def _from_base_indices(self, base_indices):
        return self._to_base_indices(base_indices)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dense(self):
        dense = self._data.dense
        for a in self._axes:
            dense = np.flip(dense, a)
        return dense

    def mask(self, mask):
        if not isinstance(mask, Encoding):
            mask = DenseEncoding(mask)
        mask = mask.flip(self._axes)
        return self._data.mask(mask).flip(self._axes)

    def copy(self):
        return FlippedEncoding(self._data.copy(), self._axes)

    def flip(self, axis=0):
        if isinstance(axis, np.ndarray):
            if axis.size == 1:
                axis = axis.item(),
            else:
                axis = tuple(axis)
        elif isinstance(axis, int):
            axes = axis,
        else:
            axes = tuple(axis)
        return _flipped(self, self._axes + axes)

    def _flip(self, axes):
        raise RuntimeError('Should not be here')


def _flipped(encoding, axes):
    if not hasattr(axes, '__iter__'):
        axes = axes,
    unique_ax = set()
    ndims = encoding.ndims
    axes = tuple(a + ndims if a < 0 else a for a in axes)
    for a in axes:
        if a in unique_ax:
            unique_ax.remove(a)
        else:
            unique_ax.add(a)
    if len(unique_ax) == 0:
        return encoding
    else:
        return encoding._flip(tuple(sorted(unique_ax)))


def _transposed(encoding, perm):
    ndims = encoding.ndims
    perm = tuple(p + ndims if p < 0 else p for p in perm)
    if np.all(np.arange(ndims) == perm):
        return encoding
    else:
        return encoding._transpose(perm)
