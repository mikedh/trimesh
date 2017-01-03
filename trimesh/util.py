'''
trimesh.util: utility functions

Only imports from numpy and the standard library are allowed in this file.

Other libraries may be included but they must be wrapped in try/except blocks
'''

import numpy as np
import logging
import hashlib
import base64
import time
import json
import zlib

from collections import defaultdict, deque
from sys import version_info
from functools import wraps
from copy import deepcopy

_PY3 = version_info.major >= 3
if _PY3:
    basestring = str
    from io import BytesIO, StringIO
else:
    from StringIO import StringIO
    
log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

# included here so util has only standard library imports
_TOL_ZERO = 1e-12

def unitize(points, check_valid=False):
    '''
    Turn a list of vectors into a list of unit vectors.

    Arguments
    ---------
    points:       (n,m) or (j) input array of vectors.
                  For 1D arrays, points is treated as a single vector
                  For 2D arrays, each row is treated as a vector
    check_valid:  boolean, if True enables valid output and checking

    Returns
    ---------
    unit_vectors: (n,m) or (j) length array of unit vectors

    valid:        (n) boolean array, output only if check_valid.
                   True for all valid (nonzero length) vectors, thus m=sum(valid)
    '''
    points = np.asanyarray(points)
    axis = len(points.shape) - 1
    length = np.sum(points ** 2, axis=axis) ** .5

    if is_sequence(length):
        length[np.isnan(length)] = 0.0

    if check_valid:
        valid = np.greater(length, _TOL_ZERO)
        if axis == 1:
            unit_vectors = (points[valid].T / length[valid]).T
        elif len(points.shape) == 1 and valid:
            unit_vectors = points / length
        else:
            unit_vectors = np.array([])
        return unit_vectors, valid
    else:
        unit_vectors = (points.T / length).T
    return unit_vectors


def euclidean(a, b):
    '''
    Euclidean distance between vectors a and b
    '''
    return np.sum((np.array(a) - b)**2) ** .5


def is_file(obj):
    return hasattr(obj, 'read')


def is_string(obj):
    return isinstance(obj, basestring)


def is_dict(obj):
    return isinstance(obj, dict)


def is_none(obj):
    if obj is None:
        return True
    if (is_sequence(obj) and
        len(obj) == 1 and
            obj[0] is None):
        return True
    return False


def is_sequence(obj):
    '''
    Returns True if obj is a sequence.
    '''
    seq = (not hasattr(obj, "strip") and
           hasattr(obj, "__getitem__") or
           hasattr(obj, "__iter__"))

    seq = seq and not isinstance(obj, dict)
    seq = seq and not isinstance(obj, set)

    # numpy sometimes returns objects that are single float64 values
    # but sure look like sequences, so we check the shape
    if hasattr(obj, 'shape'):
        seq = seq and obj.shape != ()
    return seq


def is_shape(obj, shape):
    '''
    Compare the shape of a numpy.ndarray to a target shape,
    with any value less than zero being considered a wildcard

    Note that if a list- like object is passed that is not a numpy
    array, this function will not convert it and will return False.

    Arguments
    ---------
    obj: np.ndarray to check the shape of
    shape: list or tuple of shape.
           Any negative term will be considered a wildcard
           Any tuple term will be evaluated as an OR

    Returns
    ---------
    shape_ok: bool, True if shape of obj matches query shape

    Examples
    ------------------------
    In [1]: a = np.random.random((100,3))

    In [2]: a.shape
    Out[2]: (100, 3)

    In [3]: trimesh.util.is_shape(a, (-1,3))
    Out[3]: True

    In [4]: trimesh.util.is_shape(a, (-1,3,5))
    Out[4]: False

    In [5]: trimesh.util.is_shape(a, (100,-1))
    Out[5]: True

    In [6]: trimesh.util.is_shape(a, (-1,(3,4)))
    Out[6]: True

    In [7]: trimesh.util.is_shape(a, (-1,(4,5)))
    Out[7]: False
    '''

    if (not hasattr(obj, 'shape') or
            len(obj.shape) != len(shape)):
        return False

    for i, target in zip(obj.shape, shape):
        # check if current field has multiple acceptable values
        if is_sequence(target):
            if i in target:
                continue
            else:
                return False
        # check if current field is a wildcard
        if target < 0:
            if i == 0:
                return False
            else:
                continue
        # since we have a single target and a single value,
        # if they are not equal we have an answer
        if target != i:
            return False

    # since none of the checks failed, the two shapes are the same
    return True


def make_sequence(obj):
    '''
    Given an object, if it is a sequence return, otherwise
    add it to a length 1 sequence and return.

    Useful for wrapping functions which sometimes return single
    objects and other times return lists of objects.
    '''
    if is_sequence(obj):
        return np.array(list(obj))
    else:
        return np.array([obj])


def vector_hemisphere(vectors):
    '''
    For a set of 3D vectors alter the sign so they are all in the upper
    hemisphere.

    If the vector lies on the plane, all vectors with negative Y will be reversed.
    If the vector has a zero Z and Y value, vectors with a negative X value
    will be reversed

    Arguments
    ----------
    vectors: (n,3) float, set of vectors

    Returns
    ----------
    oriented: (n,3) float, set of vectors with same magnitude but all
                           pointing in the same hemisphere.

    '''
    vectors = np.asanyarray(vectors, dtype=np.float64)
    if not is_shape(vectors, (-1, 3)):
        raise ValueError('Vectors must be (n,3)!')

    neg = vectors < -_TOL_ZERO
    zero = np.logical_not(np.logical_or(neg, vectors > _TOL_ZERO))

    # move all                          negative Z to positive
    # then for zero Z vectors, move all negative Y to positive
    # then for zero Y vectors, move all negative X to positive

    signs = np.ones(len(vectors), dtype=np.float64)

    # all vectors with negative Z values
    signs[neg[:, 2]] = -1.0
    # all on-plane vectors with negative Y values
    signs[np.logical_and(zero[:, 2], neg[:, 1])] = -1.0
    # all on-plane vectors with zero Y values and negative X values
    signs[
        np.logical_and(
            np.logical_and(
                zero[
                    :, 2], zero[
                    :, 1]), neg[
                        :, 0])] = -1.0

    oriented = vectors * signs.reshape((-1, 1))
    return oriented


def vector_to_spherical(cartesian):
    '''
    Convert a set of cartesian points to (n,2) spherical vectors
    '''
    cartesian = np.asanyarray(cartesian, dtype=np.float64)
    if not is_shape(cartesian, (-1, 3)):
        raise ValueError('Cartesian points must be (n,3)!')

    x, y, z = unitize(cartesian).T
    spherical = np.column_stack((np.arctan2(y, x),
                                 np.arccos(z)))
    return spherical


def spherical_to_vector(spherical):
    '''
    Convert a set of (n,2) spherical vectors to (n,3) vectors
    '''
    spherical = np.asanyarray(spherical, dtype=np.float64)
    if not is_shape(spherical, (-1, 2)):
        raise ValueError(
            'Spherical vectors must be passed as an (n,2) set of angles!')

    theta, phi = spherical.T
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    vectors = np.column_stack((ct * sp,
                               st * sp,
                               cp))
    return vectors


def diagonal_dot(a, b):
    '''
    Dot product by row of a and b.

    Same as np.diag(np.dot(a, b.T)) but without the monstrous
    intermediate matrix.
    '''
    result = (np.array(a) * b).sum(axis=1)
    return result


def three_dimensionalize(points, return_2D=True):
    '''
    Given a set of (n,2) or (n,3) points, return them as (n,3) points

    Arguments
    ----------
    points:    (n, 2) or (n,3) points
    return_2D: boolean flag

    Returns
    ----------
    if return_2D:
        is_2D: boolean, True if points were (n,2)
        points: (n,3) set of points
    else:
        points: (n,3) set of points
    '''
    points = np.asanyarray(points)
    shape = points.shape

    if len(shape) != 2:
        raise ValueError('Points must be 2D array!')

    if shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
        is_2D = True
    elif shape[1] == 3:
        is_2D = False
    else:
        raise ValueError('Points must be (n,2) or (n,3)!')

    if return_2D:
        return is_2D, points
    return points


def grid_arange_2D(bounds, step):
    '''
    Return a 2D grid with specified spacing

    Arguments
    ---------
    bounds: (2,2) list of [[minx, miny], [maxx, maxy]]
    step:   float, separation between points

    Returns
    -------
    grid: (n, 2) list of 2D points
    '''
    bounds = np.asanyarray(bounds)
    x_grid = np.arange(*bounds[:, 0], step=step)
    y_grid = np.arange(*bounds[:, 1], step=step)
    grid = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1, 2))
    return grid


def grid_linspace_2D(bounds, count):
    '''
    Return a count*count 2D grid

    Arguments
    ---------
    bounds: (2,2) list of [[minx, miny], [maxx, maxy]]
    count:  int, number of elements on a side

    Returns
    -------
    grid: (count**2, 2) list of 2D points
    '''
    bounds = np.asanyarray(bounds)
    x_grid = np.linspace(*bounds[:, 0], num=count)
    y_grid = np.linspace(*bounds[:, 1], num=count)
    grid = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1, 2))
    return grid


def replace_references(data, reference_dict):
    # Replace references in place
    view = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view


def multi_dict(pairs):
    '''
    Given a set of key value pairs, create a dictionary.
    If a key occurs multiple times, stack the values into an array.

    Can be called like the regular dict(pairs) constructor

    Arguments
    ----------
    pairs: (n,2) array of key, value pairs

    Returns
    ----------
    result: dict, with all values stored (rather than last with regular dict)

    '''
    result = defaultdict(list)
    for k, v in pairs:
        result[k].append(v)
    return result


def tolist_dict(data):
    def tolist(item):
        if hasattr(item, 'tolist'):
            return item.tolist()
        else:
            return item
    result = {k: tolist(v) for k, v in data.items()}
    return result


def is_binary_file(file_obj):
    '''
    Returns True if file has non-ASCII characters (> 0x7F, or 127)
    Should work in both Python 2 and 3
    '''
    start = file_obj.tell()
    fbytes = file_obj.read(1024)
    file_obj.seek(start)
    is_str = isinstance(fbytes, str)
    for fbyte in fbytes:
        if is_str:
            code = ord(fbyte)
        else:
            code = fbyte
        if code > 127:
            return True
    return False


def distance_to_end(file_obj):
    '''
    For an open file object how far is it to the end

    Arguments
    ----------
    file_obj: open file- like object

    Returns
    ----------
    distance: int, bytes to end of file
    '''
    position_current = file_obj.tell()
    file_obj.seek(0, 2)
    position_end = file_obj.tell()
    file_obj.seek(position_current)
    distance = position_end - position_current
    return distance


def decimal_to_digits(decimal, min_digits=None):
    '''
    Return the number of digits to the first nonzero decimal.

    Arguments
    -----------
    decimal:    float
    min_digits: int, minumum number of digits to return

    Returns
    -----------

    digits: int, number of digits to the first nonzero decimal
    '''
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return digits


def md5_object(obj):
    '''
    If an object is hashable, return the string of the MD5.

    Arguments
    -----------
    obj: object

    Returns
    ----------
    md5: str, MD5 hash
    '''
    hasher = hashlib.md5()
    hasher.update(obj)
    md5 = hasher.hexdigest()
    return md5


def md5_array(array, digits=5):
    '''
    Take the MD5 of an array when considering the specified number of digits.

    Arguments
    ---------
    array:  numpy array
    digits: int, number of digits to account for in the MD5

    Returns
    ---------
    md5: str, md5 hash of input
    '''
    digits = int(digits)
    array = np.asanyarray(array, dtype=np.float64).reshape(-1)
    as_int = (array * 10 ** digits).astype(np.int64)
    md5 = md5_object(as_int.tostring(order='C'))
    return md5


def attach_to_log(log_level=logging.DEBUG,
                  blacklist=['TerminalIPythonApp', 'PYREADLINE']):
    '''
    Attach a stream handler to all loggers.
    '''
    try:
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            ("%(log_color)s%(levelname)-8s%(reset)s " +
             "%(filename)17s:%(lineno)-4s  %(blue)4s%(message)s"),
            datefmt=None,
            reset=True,
            log_colors={'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red'})
    except ImportError:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
            "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(log_level)

    for logger in logging.Logger.manager.loggerDict.values():
        if (logger.__class__.__name__ != 'Logger' or
                logger.name in blacklist):
            continue
        logger.addHandler(handler_stream)
        logger.setLevel(log_level)
    np.set_printoptions(precision=5, suppress=True)


def tracked_array(array, dtype=None):
    '''
    Properly subclass a numpy ndarray to track changes.
    '''
    result = np.ascontiguousarray(array).view(TrackedArray)
    if dtype is None:
        return result
    return result.astype(dtype)


class TrackedArray(np.ndarray):
    '''
    Track changes in a numpy ndarray.

    Methods
    ----------
    md5: returns hexadecimal string of md5 of array
    crc: returns int zlib.adler32 checksum of array
    '''

    def __array_finalize__(self, obj):
        '''
        Sets a modified flag on every TrackedArray
        This flag will be set on every change, as well as during copies
        and certain types of slicing.
        '''
        self._modified_md5 = True
        self._modified_crc = True
        if isinstance(obj, type(self)):
            obj._modified_md5 = True
            obj._modified_crc = True

    def md5(self):
        '''
        Return an MD5 hash of the current array in hexadecimal string form.

        This is quite fast; on a modern i7 desktop a (1000000,3) floating point
        array was hashed reliably in .03 seconds.

        This is only recomputed if a modified flag is set which may have false
        positives (forcing an unnecessary recompute) but will not have false
        negatives which would return an incorrect hash.
        '''

        if self._modified_md5 or not hasattr(self, '_hashed_md5'):
            self._hashed_md5 = md5_object(self)  # str(self.adler32()) #
        self._modified_md5 = False
        return self._hashed_md5

    def crc(self):
        '''
        Return a zlib adler32 checksum of the current data.
        '''
        if self._modified_crc or not hasattr(self, '_hashed_crc'):
            self._hashed_crc = zlib.adler32(self) & 0xffffffff
        self._modified_crc = False
        return self._hashed_crc

    def __hash__(self):
        '''
        Hash is required to return an int, so we convert the hex string to int.
        '''
        return int(self.md5(), 16)

    def __setitem__(self, i, y):
        self._modified_md5 = True
        self._modified_crc = True
        super(self.__class__, self).__setitem__(i, y)

    def __setslice__(self, i, j, y):
        self._modified_md5 = True
        self._modified_crc = True
        super(self.__class__, self).__setslice__(i, j, y)


def cache_decorator(function):
    @wraps(function)
    def get_cached(*args, **kwargs):
        self = args[0]
        name = function.__name__
        if not (name in self._cache):
            tic = time.time()
            self._cache[name] = function(*args, **kwargs)
            toc = time.time()
            log.debug('%s was not in cache, executed in %.6f',
                      name,
                      toc - tic)
        return self._cache[name]
    return property(get_cached)


class Cache:
    '''
    Class to cache values until an id function changes.
    '''

    def __init__(self, id_function=None):
        if id_function is None:
            self._id_function = lambda: None
        else:
            self._id_function = id_function
        self.id_current = None
        self._lock = 0
        self.cache = {}

    def get(self, key):
        '''
        Get a key from the cache.

        If the key is unavailable or the cache has been invalidated returns None.
        '''
        self.verify()
        if key in self.cache:
            return self.cache[key]
        return None

    def verify(self):
        '''
        Verify that the cached values are still for the same value of id_function,
        and delete all stored items if the value of id_function has changed.
        '''
        id_new = self._id_function()
        if (self._lock == 0) and (id_new != self.id_current):
            if len(self.cache) > 0:
                log.debug('%d items cleared from cache: %s',
                          len(self.cache),
                          str(list(self.cache.keys())))
            self.clear()
            self.id_set()

    def clear(self, exclude=None):
        '''
        Remove all elements in the cache.
        '''
        if exclude is None:
            self.cache = {}
        else:
            self.cache = {k: v for k, v in self.cache.items() if k in exclude}

    def update(self, items):
        '''
        Update the cache with a set of key, value pairs without checking id_function.
        '''
        self.cache.update(items)
        self.id_set()

    def id_set(self):
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

    def __enter__(self):
        self._lock += 1

    def __exit__(self, *args):
        self._lock -= 1
        self.id_current = self._id_function()


class DataStore:

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
        if len(self.data) == 0:
            return True
        for v in self.data.values():
            if is_sequence(v):
                if len(v) > 0:
                    return False
            else:
                if bool(np.isreal(v)):
                    return False
        return True

    def clear(self):
        self.data = {}

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            return None

    def __setitem__(self, key, data):
        self.data[key] = tracked_array(data)

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
        md5 = ''
        for key in np.sort(list(self.data.keys())):
            md5 += self.data[key].md5()
        return md5

    def crc(self):
        crc_all = np.array([i.crc() for i in self.data.values()],
                           dtype=np.int64)
        crc = zlib.adler32(crc_all) & 0xffffffff
        return crc


def stack_lines(indices):
    '''
    Stack a list of values that represent a polyline into
    individual line segments with duplicated consecutive values.

    Arguments
    ----------
    indices: sequence of items

    Returns
    ---------
    stacked: (n,2) set of items

    In [1]: trimesh.util.stack_lines([0,1,2])
    Out[1]:
    array([[0, 1],
           [1, 2]])

    In [2]: trimesh.util.stack_lines([0,1,2,4,5])
    Out[2]:
    array([[0, 1],
           [1, 2],
           [2, 4],
           [4, 5]])

    In [3]: trimesh.util.stack_lines([[0,0],[1,1],[2,2], [3,3]])
    Out[3]:
    array([[0, 0],
           [1, 1],
           [1, 1],
           [2, 2],
           [2, 2],
           [3, 3]])

    '''
    indices = np.asanyarray(indices)
    if is_sequence(indices[0]):
        shape = (-1, len(indices[0]))
    else:
        shape = (-1, 2)
    return np.column_stack((indices[:-1],
                            indices[1:])).reshape(shape)


def append_faces(vertices_seq, faces_seq):
    '''
    Given a sequence of zero- indexed faces and vertices,
    combine them into a single (n,3) list of faces and (m,3) vertices

    Arguments
    ---------
    vertices_seq: (n) sequence of (m,d) vertex arrays
    faces_seq     (n) sequence of (p,j) faces, zero indexed
                  and referencing their counterpoint vertices

    '''
    vertices_len = np.array([len(i) for i in vertices_seq])
    face_offset = np.append(0, np.cumsum(vertices_len)[:-1])

    for offset, faces in zip(face_offset, faces_seq):
        faces += offset

    vertices = np.vstack(vertices_seq)
    faces = np.vstack(faces_seq)

    return vertices, faces


def array_to_string(array):
    array = np.asanyarray(array)
    if not (len(array.shape) in [1, 2]):
        raise ValueError('array_to_string is only for 1D and 2D arrays!')

    dump = json.dumps(array.tolist())
    as_str = dump.replace(
        '], ',
        '\n').replace(
        ']',
        '\n').replace(
            '[',
            '').replace(
                ',',
        '')
    as_str = as_str.strip()
    return as_str


def array_to_encoded(array, dtype=None, encoding='base64'):
    '''
    Export a numpy array to a compact serializable dictionary.

    Arguments
    ---------
    array: numpy array
    dtype: optional, what dtype should array be encoded with.
    encoding: str, 'base64' or 'binary'

    Returns
    ---------
    encoded: dict with keys:
                 dtype: string of dtype
                 shape: int tuple of shape
                 base64: base64 encoded string of flat array
    '''
    array = np.asanyarray(array)
    shape = array.shape
    # ravel also forces contiguous
    flat = np.ravel(array)
    if dtype is None:
        dtype = array.dtype

    encoded = {'dtype': np.dtype(dtype).str,
               'shape': shape}
    if encoding in ['base64', 'dict64']:
        packed = base64.b64encode(flat.astype(dtype).tostring())
        if hasattr(packed, 'decode'):
            packed = packed.decode('utf-8')
        encoded['base64'] = packed
    elif encoding == 'binary':
        encoded['binary'] = array.tostring(order='C')
    else:
        raise ValueError('encoding {} is not available!'.format(encoding))
    return encoded


def decode_keys(store, encoding='utf-8'):
    '''
    If a dictionary has keys that are bytes encode them (utf-8 default)

    Arguments
    ---------
    store: dict

    Returns
    ---------
    store: dict, with same data and if keys were bytes they have been encoded
    '''
    keys = store.keys()
    for key in keys:
        if hasattr(key, 'decode'):
            decoded = key.decode(encoding)
            if key != decoded:
                store[key.decode(encoding)] = store[key]
                store.pop(key)
    return store


def encoded_to_array(encoded):
    '''
    Turn a dictionary with base64 encoded strings back into a numpy array.

    Arguments
    ----------
    encoded: dict with keys:
                 dtype: string of dtype
                 shape: int tuple of shape
                 base64: base64 encoded string of flat array
                 binary:  decode result coming from numpy.tostring
    Returns
    ----------
    array: numpy array
    '''
    if not is_dict(encoded):
        if is_sequence(encoded):
            as_array = np.asanyarray(encoded)
            return as_array
        else:
            raise ValueError('Unable to extract numpy array from input')

    encoded = decode_keys(encoded)

    
    dtype = np.dtype(encoded['dtype'])
    if 'base64' in encoded:
        array = np.fromstring(base64.b64decode(encoded['base64']),
                              dtype)
    elif 'binary' in encoded:
        array = np.fromstring(encoded['binary'],
                              dtype=dtype)
    if 'shape' in encoded:
        array = array.reshape(encoded['shape'])
    return array


def is_instance_named(obj, name):
    '''
    Given an object, if it is a member of the class 'name',
    or a subclass of 'name', return True.

    Arguments
    ---------
    obj: instance of a class
    name: string

    Returns
    ---------
    bool, whether the object is a member of the named class
    '''
    try:
        type_named(obj, name)
        return True
    except ValueError:
        return False


def type_bases(obj, depth=4):
    '''
    Return the bases of the object passed.
    '''
    bases = deque([list(obj.__class__.__bases__)])
    for i in range(depth):
        bases.append([i.__base__ for i in bases[-1] if i is not None])
    try:
        bases = np.hstack(bases)
    except IndexError:
        bases = []
    # we do the hasattr as None/NoneType can be in the list of bases
    bases = [i for i in bases if hasattr(i, '__name__')]
    return np.array(bases)


def type_named(obj, name):
    '''
    Similar to the type() builtin, but looks in class bases for named instance.

    Arguments
    ----------
    obj: object to look for class of
    name : str, name of class

    Returns
    ----------
    named class, or None
    '''
    # if obj is a member of the named class, return True
    name = str(name)
    if obj.__class__.__name__ == name:
        return obj.__class__
    for base in type_bases(obj):
        if base.__name__ == name:
            return base
    raise ValueError('Unable to extract class of name ' + name)


def concatenate(a, b):
    '''
    Concatenate two meshes.

    Arguments
    ----------
    a: Trimesh object
    b: Trimesh object

    Returns
    ----------
    result: Trimesh object containing all faces of a and b
    '''
    # Extract the trimesh type to avoid a circular import,
    # and assert that both inputs are Trimesh objects
    trimesh_type = type_named(a, 'Trimesh')
    trimesh_type = type_named(b, 'Trimesh')

    new_normals = np.vstack((a.face_normals, b.face_normals))
    new_faces = np.vstack((a.faces, (b.faces + len(a.vertices))))
    new_vertices = np.vstack((a.vertices, b.vertices))
    new_visual = a.visual.union(b.visual)
    result = trimesh_type(vertices=new_vertices,
                          faces=new_faces,
                          face_normals=new_normals,
                          visual=new_visual,
                          process=False)
    # result._cache.id_set()
    # result.visual._cache.id_set()

    return result


def submesh(mesh,
            faces_sequence,
            only_watertight=False,
            append=False):
    '''
    Return a subset of a mesh.

    Arguments
    ----------
    mesh: Trimesh object
    faces_sequence: sequence of face indices from mesh
    only_watertight: only return submeshes which are watertight.
    append: return a single mesh which has the faces specified appended.
            if this flag is set, only_watertight is ignored

    Returns
    ---------
    if append: Trimesh object
    else:      list of Trimesh objects
    '''
    # evaluate generators so we can escape early
    faces_sequence = list(faces_sequence)

    if len(faces_sequence) == 0:
        return []

    # check to make sure we're not doing a whole bunch of work
    # to deliver a subset which ends up as the whole mesh
    if len(faces_sequence[0]) == len(mesh.faces):
        all_faces = np.array_equal(np.sort(faces_sequence),
                                   np.arange(len(faces_sequence)))
        if all_faces:
            log.debug(
                'Subset of entire mesh requested, returning copy of original')
            return mesh.copy()

    # avoid nuking the cache on the original mesh
    original_faces = mesh.faces.view(np.ndarray)
    original_vertices = mesh.vertices.view(np.ndarray)

    faces = deque()
    vertices = deque()
    normals = deque()
    visuals = deque()

    # for reindexing faces
    mask = np.arange(len(original_vertices))

    for faces_index in faces_sequence:
        # sanitize indices in case they are coming in as a set or tuple
        faces_index = np.array(list(faces_index))
        if len(faces_index) == 0:
            continue
        faces_current = original_faces[faces_index]
        unique = np.unique(faces_current.reshape(-1))

        # redefine face indices from zero
        mask[unique] = np.arange(len(unique))

        normals.append(mesh.face_normals[faces_index])
        faces.append(mask[faces_current])
        vertices.append(original_vertices[unique])
        visuals.extend(mesh.visual.subsets([faces_index]))
    # we use type(mesh) rather than importing Trimesh from base
    # as this causes a circular import
    trimesh_type = type_named(mesh, 'Trimesh')
    if append:
        visuals = np.array(visuals)
        vertices, faces = append_faces(vertices, faces)
        appended = trimesh_type(vertices=vertices,
                                faces=faces,
                                face_normals=np.vstack(normals),
                                visual=visuals[0].union(visuals[1:]),
                                process=False)
        return appended
    result = [trimesh_type(vertices=v,
                           faces=f,
                           face_normals=n,
                           visual=c,
                           metadata=mesh.metadata,
                           process=False) for v, f, n, c in zip(vertices,
                                                                faces,
                                                                normals,
                                                                visuals)]
    result = np.array(result)
    if len(result) > 0 and only_watertight:
        watertight = np.array(
            [i.fill_holes() and len(i.faces) > 4 for i in result])
        result = result[watertight]
    return result


def zero_pad(data, count, right=True):
    '''
    Arguments
    --------
    data: (n) length 1D array
    count: int

    Returns
    --------
    padded: (count) length 1D array if (n < count), otherwise length (n)
    '''
    if len(data) == 0:
        return np.zeros(count)
    elif len(data) < count:
        padded = np.zeros(count)
        if right:
            padded[-len(data):] = data
        else:
            padded[:len(data)] = data
        return padded
    else:
        return np.asanyarray(data)


def format_json(data, digits=6):
    '''
    Function to turn a 1D float array into a json string

    The built in json library doesn't have a good way of setting the
    precision of floating point numbers.

    Arguments
    ----------
    data: (n,) float array
    digits: int, number of digits of floating point numbers to include

    Returns
    ----------
    as_json: string, data formatted into a JSON- parsable string
    '''
    format_str = '.' + str(int(digits)) + 'f'
    as_json = '[' + ','.join(map(lambda o: format(o, format_str), data)) + ']'
    return as_json


class Words:
    '''
    A class to contain a list of words, such as the english language.
    The primary purpose is to create random keyphrases to be used to name
    things without resorting to giant hash strings.
    '''

    def __init__(self, file_name='/usr/share/dict/words', words=None):
        if words is None:
            self.words = np.loadtxt(file_name, dtype=str)
        else:
            self.words = np.array(words, dtype=str)

        self.words_simple = np.array([i.lower()
                                      for i in self.words if str.isalpha(i)])
        if len(self.words) == 0:
            log.warning('No words available!')

    def random_phrase(self, length=2, delimiter='-'):
        '''
        Create a random phrase using words containing only charecters.

        Arguments
        ----------
        length:    int, how many words in phrase
        delimiter: str, what to separate words with

        Returns
        ----------
        phrase: str, length words separated by delimiter

        Examples
        ----------
        In [1]: w = trimesh.util.Words()
        In [2]: for i in range(10): print w.random_phrase()
          ventilate-hindsight
          federating-flyover
          maltreat-patchiness
          puppets-remonstrated
          yoghourts-prut
          inventory-clench
          uncouple-bracket
          hipped-croupier
          puller-demesne
          phenomenally-hairs
        '''
        result = str(delimiter).join(np.random.choice(self.words_simple,
                                                      length))
        return result


def convert_like(item, like):
    '''
    Convert an item to have the dtype of another item

    Arguments
    ----------
    item: item to be converted
    like: object with target dtype. If None, item is returned unmodified

    Returns
    --------
    result: item, but in dtype of like
    '''
    if isinstance(like, np.ndarray):
        return np.asanyarray(item, dtype=like.dtype)
    
    if isinstance(item, like.__class__) or is_none(like):
        return item

    if (is_sequence(item) and
        len(item) == 1 and
        isinstance(item[0], like.__class__)):
        return item[0]
    
    item = like.__class__(item)
    return item

def round_sigfig(value, sigfig=1):
    '''
    Round a single value to a specified number of signifigant figures.

    Arguments
    ----------
    values: float, value to be rounded
    sigfig: int, number of signifigant figures to reduce to


    Returns
    ----------
    rounded: values, but rounded to the specified number of signifigant figures


    Example
    ----------
    In [1]: trimesh.util.round_sigfig(-232453.00014045456, 1)
    Out[1]: -200000.0

    In [2]: trimesh.util.round_sigfig(.00014045456, 1)
    Out[2]: 0.0001

    In [3]: trimesh.util.round_sigfig(.00014045456, 4)
    Out[3]: 0.0001405
    '''
    sigfig = int(sigfig)
    value = float(value)
    digits = int(np.floor(np.log10(np.abs(value))))
    rounded = np.round(value, sigfig - digits - 1)
    return rounded


def bounds_tree(bounds):
    '''
    Given a set of axis aligned bounds, create an r-tree for broad- phase
    collision detection

    Arguments
    ---------
    bounds: (n, dimension*2) list of non- interleaved bounds
             for a 2D bounds tree:
             [(minx, miny, maxx, maxy), ...]

    Returns
    ---------
    tree: Rtree object
    '''
    bounds = np.asanyarray(deepcopy(bounds), dtype=np.float64)
    if len(bounds.shape) != 2:
        raise ValueError('Bounds must be (n,dimension*2)!')

    dimension = bounds.shape[1]
    if (dimension % 2) != 0:
        raise ValueError('Bounds must be (n,dimension*2)!')
    dimension = int(dimension / 2)

    import rtree
    # some versions of rtree screw up indexes on stream loading
    # do a test here so we know if we are free to use stream loading
    # or if we have to do a loop to insert things which is 5x slower
    rtree_test = rtree.index.Index([(1564, [0, 0, 0, 10, 10, 10], None)],
                                   properties=rtree.index.Property(dimension=3))
    rtree_stream_ok = next(rtree_test.intersection([1, 1, 1, 2, 2, 2])) == 1564

    properties = rtree.index.Property(dimension=dimension)
    if rtree_stream_ok:
        # stream load was verified working on inport above
        tree = rtree.index.Index(zip(np.arange(len(bounds)),
                                     bounds,
                                     [None] * len(bounds)),
                                 properties=properties)
    else:
        # in some rtree versions stream loading goofs the index
        log.warning('rtree stream loading broken! Try upgrading rtree!')
        tree = rtree.index.Index(properties=properties)
        for i, b in enumerate(bounds):
            tree.insert(i, b)
    return tree

def wrap_as_stream(item):
    '''
    Wrap a string or bytes object as a file object

    Arguments
    ----------
    item: str, bytes: item to be wrapped

    Returns
    ---------
    wrapped: file-like object
    '''
    if not _PY3:
        return StringIO(item)
    if isinstance(item, str):
        return StringIO(item)
    elif isinstance(item, bytes):
        return BytesIO(item)
    raise ValueError('Not a wrappable item!')
