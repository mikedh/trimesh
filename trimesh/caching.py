"""
caching.py
-----------

Functions and classes that help with tracking changes in ndarrays
and clearing cached values based on those changes.
"""

import numpy as np

import hashlib
import zlib

from .constants import log
from .util import is_sequence

try:
    # xxhash is roughly 5x faster than adler32 but is only
    # packaged in easy wheels on linux (`pip install xxhash`)
    # so keep it as a soft dependency
    import xxhash
    hasX = True
except ImportError:
    hasX = False


def tracked_array(array, dtype=None):
    """
    Properly subclass a numpy ndarray to track changes.

    Avoids some pitfalls of subclassing by forcing contiguous
    arrays, and does a view into a TrackedArray.

    Parameters
    ------------
    array: array- like object to be turned into a TrackedArray
    dtype: which dtype to use for the array

    Returns
    ------------
    tracked: TrackedArray, of input array data
    """
    tracked = np.ascontiguousarray(array,
                                   dtype=dtype).view(TrackedArray)
    assert tracked.flags['C_CONTIGUOUS']

    return tracked


class TrackedArray(np.ndarray):
    """
    Track changes in a numpy ndarray.

    General method is to agressivly set 'modified' flags
    on operations which might alter the array.

    We store boolean modified flag for each hash type to
    make checks fast even for queries of different hashes.

    This will force a recompute of a checksum value in some
    cases where nothing has changed but we don't ever want to
    return a checksum that doesn't reflect the data.

    Methods
    ----------
    md5:       returns str, hexadecimal MD5 of array
    crc:       returns int, zlib crc32/adler32 checksum
    fast_hash: returns int, CRC or xxhash.xx64
    """

    def __array_finalize__(self, obj):
        """
        Sets a modified flag on every TrackedArray
        This flag will be set on every change as well as
        during copies and certain types of slicing.
        """
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        if isinstance(obj, type(self)):
            obj._modified_c = True
            obj._modified_m = True
            obj._modified_x = True

    def md5(self):
        """
        Return an MD5 hash of the current array.

        Returns
        -----------
        md5: str, hexadecimal MD5 of the array
        """
        if self._modified_m or not hasattr(self, '_hashed_md5'):
            if self.flags['C_CONTIGUOUS']:
                hasher = hashlib.md5(self)
                self._hashed_md5 = hasher.hexdigest()
            else:
                # the case where we have sliced our nice
                # contiguous array into a non- contiguous block
                # for example (note slice *after* track operation):
                # t = util.tracked_array(np.random.random(10))[::-1]
                contiguous = np.ascontiguousarray(self)
                hasher = hashlib.md5(contiguous)
                self._hashed_md5 = hasher.hexdigest()
        self._modified_m = False
        return self._hashed_md5

    def crc(self):
        """
        A zlib.crc32 or zlib.adler32 checksum
        of the current data.

        Returns
        -----------
        crc: int, checksum from zlib.crc32 or zlib.adler32
        """
        if self._modified_c or not hasattr(self, '_hashed_crc'):
            if self.flags['C_CONTIGUOUS']:
                self._hashed_crc = crc32(self)
            else:
                # the case where we have sliced our nice
                # contiguous array into a non- contiguous block
                # for example (note slice *after* track operation):
                # t = util.tracked_array(np.random.random(10))[::-1]
                contiguous = np.ascontiguousarray(self)
                self._hashed_crc = crc32(contiguous)
        self._modified_c = False
        return self._hashed_crc

    def _xxhash(self):
        """
        An xxhash.b64 hash of the array.

        Returns
        -------------
        xx: int, xxhash.xxh64 hash of array.
        """
        # repeat the bookkeeping to get a contiguous array inside
        # the function to avoid additional function calls
        # these functions are called millions of times so everything helps
        if self._modified_x or not hasattr(self, '_hashed_xx'):
            if self.flags['C_CONTIGUOUS']:
                hasher = xxhash.xxh64(self)
                self._hashed_xx = hasher.intdigest()
            else:
                # the case where we have sliced our nice
                # contiguous array into a non- contiguous block
                # for example (note slice *after* track operation):
                # t = util.tracked_array(np.random.random(10))[::-1]
                contiguous = np.ascontiguousarray(self)
                hasher = xxhash.xxh64(contiguous)
                self._hashed_xx = hasher.intdigest()
        self._modified_x = False
        return self._hashed_xx

    def __hash__(self):
        """
        Hash is required to return an int.

        Returns
        -----------
        hash: int, result of fast_hash
        """
        return self.fast_hash()

    def __iadd__(self, *args, **kwargs):
        """
        In- place addition.

        The i* operations are in- place and modify the array,
        so we better catch all of them.
        """
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__iadd__(*args,
                                                    **kwargs)

    def __isub__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__isub__(*args,
                                                    **kwargs)

    def __imul__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__imul__(*args,
                                                    **kwargs)

    def __idiv__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__idiv__(*args,
                                                    **kwargs)

    def __itruediv__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__itruediv__(*args,
                                                        **kwargs)

    def __imatmul__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__imatmul__(*args,
                                                       **kwargs)

    def __ipow__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__ipow__(*args, **kwargs)

    def __imod__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__imod__(*args, **kwargs)

    def __ifloordiv__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__ifloordiv__(*args,
                                                         **kwargs)

    def __ilshift__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__ilshift__(*args,
                                                       **kwargs)

    def __irshift__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__irshift__(*args,
                                                       **kwargs)

    def __iand__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__iand__(*args,
                                                    **kwargs)

    def __ixor__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__ixor__(*args,
                                                    **kwargs)

    def __ior__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        return super(self.__class__, self).__ior__(*args,
                                                   **kwargs)

    def __setitem__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        super(self.__class__, self).__setitem__(*args,
                                                **kwargs)

    def __setslice__(self, *args, **kwargs):
        self._modified_c = True
        self._modified_m = True
        self._modified_x = True
        super(self.__class__, self).__setslice__(*args,
                                                 **kwargs)

    if hasX:
        # if xxhash is installed use it
        fast_hash = _xxhash
    else:
        # otherwise use our fastest CRC
        fast_hash = crc


class Cache:
    """
    Class to cache values which will be stored until the
    result of an ID function changes.
    """

    def __init__(self, id_function):
        """
        Create a cache object.

        Parameters
        ------------
        id_function: function, that returns hashable value
        """
        self._id_function = id_function

        self.id_current = self._id_function()
        self._lock = 0
        self.cache = {}

    def get(self, key):
        """
        Get a key from the cache.

        If the key is unavailable or the cache has been invalidated
        returns None.

        Parameters
        -------------
        key: hashable value

        Returns
        -------------
        value: value from cache
        """
        self.verify()
        if key in self.cache:
            return self.cache[key]
        return None

    def delete(self, key):
        """
        Remove a key from the cache.
        """
        if key in self.cache:
            self.cache.pop(key, None)

    def verify(self):
        """
        Verify that the cached values are still for the same
        value of id_function and delete all stored items if
        the value of id_function has changed.
        """
        id_new = self._id_function()
        if (self._lock == 0) and (id_new != self.id_current):
            if len(self.cache) > 0:
                log.debug('%d items cleared from cache: %s',
                          len(self.cache),
                          str(list(self.cache.keys())))
            self.clear()
            self.id_set()

    def clear(self, exclude=None):
        """
        Remove all elements in the cache.
        """
        if exclude is None:
            self.cache = {}
        else:
            self.cache = {k: v for k, v in self.cache.items()
                          if k in exclude}

    def update(self, items):
        """
        Update the cache with a set of key, value pairs without
        checking id_function.
        """
        self.cache.update(items)
        self.id_set()

    def id_set(self):
        """
        Set the current ID to the value of the ID function.
        """
        self.id_current = self._id_function()

    def set(self, key, value):
        self.verify()
        self.cache[key] = value
        return value

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __contains__(self, key):
        self.verify()
        return key in self.cache

    def __len__(self):
        self.verify()
        return len(self.cache)

    def __enter__(self):
        self._lock += 1

    def __exit__(self, *args):
        self._lock -= 1
        self.id_current = self._id_function()


class DataStore:
    """
    A class to store multiple numpy arrays and track them all
    for changes.

    Operates like a dict that only stores numpy.ndarray
    """

    def __init__(self):
        self.data = {}

    @property
    def mutable(self):
        if not hasattr(self, '_mutable'):
            self._mutable = True
        return self._mutable

    @mutable.setter
    def mutable(self, value):
        value = bool(value)
        for i in self.data.value():
            i.flags.writeable = value
        self._mutable = value

    def is_empty(self):
        """
        Is the current DataStore empty or not.

        Returns
        ----------
        empty: bool, False if there are items in the DataStore
        """
        if len(self.data) == 0:
            return True
        for v in self.data.values():
            if is_sequence(v):
                if len(v) == 0:
                    return True
                else:
                    return False
            elif bool(np.isreal(v)):
                return False
        return True

    def clear(self):
        """
        Remove all data from the DataStore.
        """
        self.data = {}

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            return np.empty((0,))

    def __setitem__(self, key, data):
        """
        Store an item in the DataStore
        """
        if hasattr(data, 'md5'):
            # don't bother to re-track TrackedArray
            self.data[key] = data
        else:
            # otherwise wrap data
            self.data[key] = tracked_array(data)

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def values(self):
        return self.data.values()

    def update(self, values):
        if not isinstance(values, dict):
            raise ValueError('Update only implemented for dicts')
        for key, value in values.items():
            self[key] = value

    def md5(self):
        """
        Get an MD5 reflecting everything in the DataStore.

        Returns
        ----------
        md5: str, MD5 in hexadecimal
        """
        hasher = hashlib.md5()
        for key in sorted(self.data.keys()):
            hasher.update(self.data[key].md5().encode('utf-8'))
        md5 = hasher.hexdigest()
        return md5

    def crc(self):
        """
        Get a CRC reflecting everything in the DataStore.

        Returns
        ----------
        crc: int, CRC of data
        """
        crc = sum(i.crc() for i in self.data.values())
        return crc

    def fast_hash(self):
        """
        Get a CRC32 or xxhash.xxh64 reflecting the DataStore.

        Returns
        ------------
        hashed: int, checksum of data
        """
        fast = sum(i.fast_hash() for i in self.data.values())
        return fast


def _fast_crc(count=50):
    """
    On certain platforms/builds zlib.adler32 is substantially
    faster than zlib.crc32, but it is not consistent across
    Windows/Linux/OSX.

    This function runs a quick check (2ms on my machines) to
    determine the fastest hashing function available in zlib.

    Parameters
    ------------
    count: int, number of repetitions to do on the speed trial

    Returns
    ----------
    crc32: function, either zlib.adler32 or zlib.crc32
    """
    import timeit

    setup = 'import numpy, zlib;'
    setup += 'd = numpy.random.random((500,3));'

    crc32 = timeit.timeit(setup=setup,
                          stmt='zlib.crc32(d)',
                          number=count)
    adler32 = timeit.timeit(setup=setup,
                            stmt='zlib.adler32(d)',
                            number=count)
    if adler32 < crc32:
        return zlib.adler32
    else:
        return zlib.crc32


# get the fastest CRC32 available on the
# current platform when trimesh is imported
crc32 = _fast_crc()
