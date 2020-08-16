"""
caching.py
-----------

Functions and classes that help with tracking changes in ndarrays
and clearing cached values based on those changes.
"""

import numpy as np

import zlib
import hashlib

from functools import wraps

from .constants import log
from .util import is_sequence

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

try:
    # xxhash is roughly 5x faster than zlib.adler32 but is only
    # packaged in easy wheels on linux (`pip install xxhash`)
    # so we keep it a soft dependency
    import xxhash
except ImportError:
    xxhash = None


def tracked_array(array, dtype=None):
    """
    Properly subclass a numpy ndarray to track changes.

    Avoids some pitfalls of subclassing by forcing contiguous
    arrays and does a view into a TrackedArray.

    Parameters
    ------------
    array : array- like object
      To be turned into a TrackedArray
    dtype : np.dtype
      Which dtype to use for the array

    Returns
    ------------
    tracked : TrackedArray
      Contains input array data
    """
    # if someone passed us None, just create an empty array
    if array is None:
        array = []
    # make sure it is contiguous then view it as our subclass
    tracked = np.ascontiguousarray(
        array, dtype=dtype).view(TrackedArray)
    # should always be contiguous here
    assert tracked.flags['C_CONTIGUOUS']

    return tracked


def cache_decorator(function):
    """
    A decorator for class methods, replaces @property
    but will store and retrieve function return values
    in object cache.

    Parameters
    ------------
    function : method
      This is used as a decorator:
      ```
      @cache_decorator
      def foo(self, things):
        return 'happy days'
      ```
    """

    # use wraps to preserve docstring
    @wraps(function)
    def get_cached(*args, **kwargs):
        """
        Only execute the function if its value isn't stored
        in cache already.
        """
        self = args[0]
        # use function name as key in cache
        name = function.__name__
        # do the dump logic ourselves to avoid
        # verifying cache twice per call
        self._cache.verify()
        # access cache dict to avoid automatic validation
        # since we already called cache.verify manually
        if name in self._cache.cache:
            # already stored so return value
            return self._cache.cache[name]
        # value not in cache so execute the function
        value = function(*args, **kwargs)
        # store the value
        if self._cache.force_immutable and hasattr(
                value, 'flags') and len(value.shape) > 0:
            value.flags.writeable = False

        self._cache.cache[name] = value

        return value

    # all cached values are also properties
    # so they can be accessed like value attributes
    # rather than functions
    return property(get_cached)


class TrackedArray(np.ndarray):
    """
    Subclass of numpy.ndarray that provides hash methods
    to track changes.

    General method is to aggressively set 'modified' flags
    on operations which might (but don't necessarily) alter
    the array, ideally we sometimes compute hashes when we
    don't need to, but we don't return wrong hashes ever.

    We store boolean modified flag for each hash type to
    make checks fast even for queries of different hashes.

    Methods
    ----------
    md5 :       str, hexadecimal MD5 of array
    crc :       int, zlib crc32/adler32 checksum
    fast_hash : int, CRC or xxhash.xx64
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

    @property
    def mutable(self):
        return self.flags['WRITEABLE']

    @mutable.setter
    def mutable(self, value):
        self.flags.writeable = value

    def md5(self):
        """
        Return an MD5 hash of the current array.

        Returns
        -----------
        md5 : str
          Hexadecimal MD5 of the array
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
        crc : int
          Checksum from zlib.crc32 or zlib.adler32
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
        xx : int
          xxhash.xxh64 hash of array.
        """
        # repeat the bookkeeping to get a contiguous array inside
        # the function to avoid additional function calls
        # these functions are called millions of times so everything helps
        if self._modified_x or not hasattr(self, '_hashed_xx'):
            if self.flags['C_CONTIGUOUS']:
                self._hashed_xx = xxhash.xxh64(self).intdigest()
            else:
                # the case where we have sliced our nice
                # contiguous array into a non- contiguous block
                # for example (note slice *after* track operation):
                # t = util.tracked_array(np.random.random(10))[::-1]
                self._hashed_xx = xxhash.xxh64(np.ascontiguousarray(self)).intdigest()
        self._modified_x = False
        return self._hashed_xx

    def __hash__(self):
        """
        Hash is required to return an int.

        Returns
        -----------
        hash : int
          Result of fast_hash
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

    if xxhash is None:
        # otherwise use our fastest CRC
        fast_hash = crc
    else:
        # if xxhash is installed use it
        fast_hash = _xxhash


class Cache(object):
    """
    Class to cache values which will be stored until the
    result of an ID function changes.
    """

    def __init__(self, id_function, force_immutable=False):
        """
        Create a cache object.

        Parameters
        ------------
        id_function : function
          Returns hashable value
        force_immutable : bool
          If set will make all numpy arrays read-only
        """
        self._id_function = id_function
        # force stored numpy arrays to have flags.writable=False
        self.force_immutable = bool(force_immutable)
        # call the id function for initial value
        self.id_current = self._id_function()
        # a counter for locks
        self._lock = 0
        # actual store for data
        self.cache = {}

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
        # if we are in a lock don't check anything
        if self._lock != 0:
            return

        # check the hash of our data
        id_new = self._id_function()

        # things changed
        if id_new != self.id_current:
            if len(self.cache) > 0:
                log.debug('%d items cleared from cache: %s',
                          len(self.cache),
                          str(list(self.cache.keys())))
            # hash changed, so dump the cache
            # do it manually rather than calling clear()
            # as we are internal logic and can avoid function calls
            self.cache = {}
            # set the id to the new data hash
            self.id_current = id_new

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

        if self.force_immutable:
            for k, v in self.cache.items():
                if hasattr(v, 'flags') and len(v.shape) > 0:
                    v.flags.writeable = False
        self.id_set()

    def id_set(self):
        """
        Set the current ID to the value of the ID function.
        """
        self.id_current = self._id_function()

    def __getitem__(self, key):
        """
        Get an item from the cache. If the item
        is not in the cache, it will return None

        Parameters
        -------------
        key : hashable
               Key in dict

        Returns
        -------------
        cached : object, or None
          Object that was stored
        """
        self.verify()
        if key in self.cache:
            return self.cache[key]
        return None

    def __setitem__(self, key, value):
        """
        Add an item to the cache.

        Parameters
        ------------
        key : hashable
          Key to reference value
        value : any
          Value to store in cache
        """
        # dumpy cache if ID function has changed
        self.verify()
        # make numpy arrays read-only if asked to
        if self.force_immutable and hasattr(value, 'flags') and len(value.shape) > 0:
            value.flags.writeable = False
        # assign data to dict
        self.cache[key] = value

        return value

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


class DataStore(Mapping):
    """
    A class to store multiple numpy arrays and track them all
    for changes.

    Operates like a dict that only stores numpy.ndarray
    """

    def __init__(self):
        self.data = {}

    def __iter__(self):
        return iter(self.data)

    def __delitem__(self, key):
        del self.data[key]

    @property
    def mutable(self):
        """
        Is data allowed to be altered or not.

        Returns
        -----------
        is_mutable : bool
          Can data be altered in the DataStore
        """
        if not hasattr(self, '_mutable'):
            return True
        return self._mutable

    @mutable.setter
    def mutable(self, value):
        """
        Is data allowed to be altered or not.

        Parameters
        ------------
        is_mutable : bool
          Should data be allowed to be altered
        """
        # make sure passed value is a bool
        is_mutable = bool(value)
        # apply the flag to any data stored
        for n, i in self.data.items():
            i.mutable = value
        # save the mutable setting
        self._mutable = is_mutable

    def is_empty(self):
        """
        Is the current DataStore empty or not.

        Returns
        ----------
        empty : bool
          False if there are items in the DataStore
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
        return self.data[key]

    def __setitem__(self, key, data):
        """
        Store an item in the DataStore
        """
        # we shouldn't allow setting on immutable datastores
        if not self.mutable:
            raise ValueError('DataStore is configured immutable!')

        if hasattr(data, 'md5'):
            # don't bother to re-track TrackedArray
            tracked = data
        else:
            # otherwise wrap data
            tracked = tracked_array(data)
        # apply our mutability setting

        if hasattr(self, '_mutable'):
            # apply our mutability setting only if it was explicitly set
            tracked.mutable = self.mutable
        # store data
        self.data[key] = tracked

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

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
        md5 : str
          MD5 of data in hexadecimal
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
        crc : int
          CRC of data
        """
        # combine with a sum of every hash
        crc = sum(i.crc() for i in self.data.values())
        return crc

    def fast_hash(self):
        """
        Get a CRC32 or xxhash.xxh64 reflecting the DataStore.

        Returns
        ------------
        hashed : int
          Checksum of data
        """
        # combine every hash
        fast = sum(i.fast_hash() for i in self.data.values())
        return fast


def _fast_crc(count=25):
    """
    On certain platforms/builds zlib.adler32 is substantially
    faster than zlib.crc32, but it is not consistent across
    Windows/Linux/OSX.

    This function runs a quick check (2ms on my machines) to
    determine the fastest hashing function available in zlib.

    Parameters
    ------------
    count : int
      Number of repetitions to do on the speed trial

    Returns
    ----------
    crc32 : function
      Either `zlib.adler32` or `zlib.crc32`
    """
    import timeit

    # create an array of random numbers
    setup = 'import numpy, zlib;'
    setup += 'd = numpy.random.random((500,3));'
    # time crc32
    crc32 = timeit.timeit(setup=setup,
                          stmt='zlib.crc32(d)',
                          number=count)
    # time adler32
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
