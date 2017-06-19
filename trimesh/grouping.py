import numpy as np
import networkx as nx

from . import util

from .constants import log, tol
from collections import deque

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    log.warning('Scipy unavailable')


def merge_vertices_hash(mesh):
    '''
    Removes duplicate vertices, based on integer hashes.
    This is roughly 20x faster than querying a KD tree in a loop
    '''
    unique, inverse = unique_rows(mesh.vertices)
    mesh.update_vertices(unique, inverse)


def replace_references(data, reference_dict):
    '''
    Replace elements in an array as per a dictionary of replacement values.

    Parameters
    ----------
    data:           numpy array
    reference_dict: dictionary of replacement value mapping, eg: {2:1, 3:1, 4:5}
    '''
    shape = np.shape(data)
    view = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view.reshape(shape)


def group(values, min_len=0, max_len=np.inf):
    '''
    Return the indices of values that are identical

    Parameters
    ----------
    values:     1D array
    min_len:    int, the shortest group allowed
                All groups will have len >= min_length
    max_len:    int, the longest group allowed
                All groups will have len <= max_length

    Returns
    ----------
    groups: sequence of indices to form groups
            IE [0,1,0,1] returns [[0,2], [1,3]]
    '''
    values = np.asanyarray(values)
    order = values.argsort()
    values = values[order]
    dupe = np.greater(np.abs(np.diff(values)), tol.zero)
    dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
    dupe_len = np.diff(np.hstack((dupe_idx, len(values))))
    dupe_ok = np.logical_and(np.greater_equal(dupe_len, min_len),
                             np.less_equal(dupe_len, max_len))
    groups = [order[i:(i + j)]
              for i, j in zip(dupe_idx[dupe_ok], dupe_len[dupe_ok])]
    groups = np.array(groups)
    return groups


def hashable_rows(data, digits=None):
    '''
    We turn our array into integers, based on the precision
    given by digits, and then put them in a hashable format.

    Parameters
    ---------
    data:    (n,m) input array
    digits:  how many digits to add to hash, if data is floating point
             If none, TOL_MERGE will be turned into a digit count and used.

    Returns
    ---------
    hashable:  (n) length array of custom data which can be sorted
                or used as hash keys
    '''
    as_int = float_to_int(data, digits)
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    hashable = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
    return hashable


def float_to_int(data, digits=None, dtype_out=np.int32):
    '''
    Given a numpy array of data represent it as integers.
    '''
    data = np.asanyarray(data)

    if data.size == 0:
        return data.astype(dtype_out)
    if digits is None:
        digits = util.decimal_to_digits(tol.merge)
    elif isinstance(digits, float) or isinstance(digits, np.float):
        digits = util.decimal_to_digits(digits)
    elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
        log.warn('Digits were passed as %s!', digits.__class__.__name__)
        raise ValueError('Digits must be None, int, or float!')

    # if data is already an integer or boolean, we're done
    if data.dtype.kind in 'ib':
        as_int = data.astype(dtype_out)
    else:
        data_max = np.abs(data).max() * 10**digits
        dtype_out = [np.int32, np.int64][int(data_max > 2**31)]
        as_int = (np.round(data, digits) * (10**digits)).astype(dtype_out)

    return as_int


def unique_ordered(data):
    '''
    Returns the same as np.unique, but ordered as per the
    first occurance of the unique value in data.

    Example
    ---------
    In [1]: a = [0, 3, 3, 4, 1, 3, 0, 3, 2, 1]

    In [2]: np.unique(a)
    Out[2]: array([0, 1, 2, 3, 4])

    In [3]: trimesh.grouping.unique_ordered(a)
    Out[3]: array([0, 3, 4, 1, 2])
    '''
    data = np.asanyarray(data)
    order = np.sort(np.unique(data, return_index=True)[1])
    result = data[order]
    return result


def merge_runs(data, digits=None):
    '''
    Merge duplicate sequential values.

    Parameters
    -----------
    data: (n,) float or int

    Returns
    --------
    merged: (m,) float or int

    Example
    ---------
    In [1]: a
    Out[1]: 
    array([-1, -1, -1,  0,  0,  1,  1,  2,  0,  
            3,  3,  4,  4,  5,  5,  6,  6,  7,  
            7,  8,  8,  9,  9,  9])

    In [2]: trimesh.grouping.merge_runs(a)
    Out[2]: array([-1,  0,  1,  2,  0,  3,  4,  5,  6,  7,  8,  9])
    '''
    data = np.asanyarray(data)
    data_int = float_to_int(data, digits=digits)
    mask = np.append(True, np.diff(data_int).astype(bool))

    result = data[mask]
    return result


def unique_float(data,
                 return_index=False,
                 return_inverse=False,
                 digits=None):
    '''
    Identical to the numpy.unique command, except evaluates floating point
    numbers, using a specified number of digits.

    If digits isn't specified, the libray default TOL_MERGE will be used.
    '''

    as_int = float_to_int(data, digits)
    _junk, unique, inverse = np.unique(as_int,
                                       return_index=True,
                                       return_inverse=True)
    if (not return_index) and (not return_inverse):
        return data[unique]
    result = [data[unique]]
    if return_index:
        result.append(unique)
    if return_inverse:
        result.append(inverse)
    return tuple(result)


def unique_rows(data, digits=None):
    '''
    Returns indices of unique rows. It will return the
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]

    Parameters
    ---------
    data: (n,m) set of floating point data
    digits: how many digits to consider for the purposes of uniqueness

    Returns
    --------
    unique:  (j) array, index in data which is a unique row
    inverse: (n) length array to reconstruct original
                 example: unique[inverse] == data
    '''
    hashes = hashable_rows(data, digits=digits)
    garbage, unique, inverse = np.unique(hashes,
                                         return_index=True,
                                         return_inverse=True)
    return unique, inverse


def unique_value_in_row(data, unique=None):
    '''
    For a 2D array of integers find the position of a value in each
    row which only occurs once. If there are more than one value per
    row which occur once, the last one is returned.

    Parameters
    ----------
    data:   (n,d) int
    unique: (m) int, list of unique values contained in data.
             speedup purposes only, generated from np.unique if not passed

    Returns
    ---------
    result: (n,d) bool, with one or zero True values per row.


    Example
    -------------------------------------
    In [0]: r = np.array([[-1,  1,  1],
                          [-1,  1, -1],
                          [-1,  1,  1],
                          [-1,  1, -1],
                          [-1,  1, -1]], dtype=np.int8)

    In [1]: unique_value_in_row(r)
    Out[1]:
           array([[ True, False, False],
                  [False,  True, False],
                  [ True, False, False],
                  [False,  True, False],
                  [False,  True, False]], dtype=bool)

    In [2]: unique_value_in_row(r).sum(axis=1)
    Out[2]: array([1, 1, 1, 1, 1])

    In [3]: r[unique_value_in_row(r)]
    Out[3]: array([-1,  1, -1,  1,  1], dtype=int8)
    '''
    if unique is None:
        unique = np.unique(data)
    data = np.asanyarray(data)
    result = np.zeros_like(data, dtype=np.bool, subok=False)
    for value in unique:
        test = np.equal(data, value)
        test_ok = test.sum(axis=1) == 1
        result[test_ok] = test[test_ok]
    return result


def group_rows(data, require_count=None, digits=None):
    '''
    Returns index groups of duplicate rows, for example:
    [[1,2], [3,4], [1,2]] will return [[0,2], [1]]

    Parameters
    ----------
    data:          (n,m) array
    require_count: only returns groups of a specified length, eg:
                   require_count =  2
                   [[1,2], [3,4], [1,2]] will return [[0,2]]

                   Note that using require_count allows numpy advanced indexing
                   to be used in place of looping and checking hashes, and as a
                   consequence is ~10x faster.

    digits:        If data is floating point, how many decimals to look at.
                   If this is None, the value in TOL_MERGE will be turned into a
                   digit count and used.

    Returns
    ----------
    groups:        List or sequence of indices from data indicating identical rows.
                   If require_count != None, shape will be (j, require_count)
                   If require_count is None, shape will be irregular (AKA a sequence)
    '''

    def group_dict():
        '''
        Simple hash table based grouping.
        The loop and appends make this rather slow on very large arrays,
        but it works on irregular groups.
        '''
        observed = dict()
        hashable = hashable_rows(data, digits=digits)
        for index, key in enumerate(hashable):
            key_string = key.tostring()
            if key_string in observed:
                observed[key_string].append(index)
            else:
                observed[key_string] = [index]
        return np.array(list(observed.values()))

    def group_slice():
        # create a representation of the rows that can be sorted
        hashable = hashable_rows(data, digits=digits)
        # record the order of the rows so we can get the original indices back
        # later
        order = np.argsort(hashable)
        # but for now, we want our hashes sorted
        hashable = hashable[order]
        # this is checking each neighbour for equality, example:
        # example: hashable = [1, 1, 1]; dupe = [0, 0]
        dupe = hashable[1:] != hashable[:-1]
        # we want the first index of a group, so we can slice from that location
        # example: hashable = [0 1 1]; dupe = [1,0]; dupe_idx = [0,1]
        dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
        # if you wanted to use this one function to deal with non- regular groups
        # you could use: np.array_split(dupe_idx)
        # this is roughly 3x slower than using the group_dict method above.
        start_ok = np.diff(
            np.hstack((dupe_idx, len(hashable)))) == require_count
        groups = np.tile(dupe_idx[start_ok].reshape((-1, 1)),
                         require_count) + np.arange(require_count)
        groups_idx = order[groups]
        if require_count == 1:
            return groups_idx.reshape(-1)
        return groups_idx

    if require_count is None:
        return group_dict()
    else:
        return group_slice()


def boolean_rows(a, b, operation=set.intersection):
    '''
    Find the rows in two arrays which occur in both rows.

    Parameters
    ---------
    a: (n,d) array
    b: (m,d) array
    operation: boolean set operation function:
               set.intersection
               set.difference
               set.union

    Returns
    --------
    shared: (p, d) array containing rows in both a and b
    '''
    a = float_to_int(a)
    b = float_to_int(b)
    shared = operation({tuple(i) for i in a}, {tuple(j) for j in b})
    shared = np.array(list(shared))
    return shared


def group_vectors(vectors,
                  angle=np.radians(10),
                  include_negative=False):
    '''
    Group vectors based on an angle tolerance, with the option to
    include negative vectors.


    '''
    if include_negative:
        vectors = util.vector_hemisphere(vectors)

    spherical = util.vector_to_spherical(vectors)
    angles, groups = group_distance(spherical, angle)
    new_vectors = util.spherical_to_vector(angles)
    return new_vectors, groups


def group_distance(values, distance):
    consumed = np.zeros(len(values), dtype=np.bool)
    tree = KDTree(values)

    # (n, d) set of values that are unique
    unique = deque()
    # (n) sequence of indicies in values
    groups = deque()

    for index, value in enumerate(values):
        if consumed[index]:
            continue
        group = np.array(tree.query_ball_point(value, distance), dtype=np.int)
        consumed[group] = True
        unique.append(np.median(values[group], axis=0))
        groups.append(group)
    return np.array(unique), np.array(groups)


def clusters(points, radius):
    '''
    Find clusters of points which have neighbours closer than radius

    Parameters
    ---------
    points: (n, d) points (of dimension d)
    radius: max distance between points in a cluster

    Returns:
    groups: (m) sequence of indices for points

    '''
    tree = KDTree(points)
    pairs = tree.query_pairs(radius)
    graph = nx.from_edgelist(pairs)
    groups = list(nx.connected_components(graph))
    return groups


def blocks(data, min_len=2, max_len=np.inf, digits=None, only_nonzero=False):
    '''
    Given an array, find the indices of contiguous blocks
    of equal values.

    Parameters
    ---------
    data:    (n) array
    min_len: int, the minimum length group to be returned
    max_len: int, the maximum length group to be retuurned
    digits:  if dealing with floats, how many digits to use
    only_nonzero: bool, only return blocks of non- zero values

    Returns
    ---------
    blocks: (m) sequence of indices referencing data
    '''
    data = float_to_int(data, digits=digits)

    # find the inflection points, or locations where the array turns
    # from True to False.
    infl = np.hstack(([0],
                      np.nonzero(np.diff(data))[0] + 1,
                      [len(data)]))
    infl_len = np.diff(infl)
    infl_ok = np.logical_and(infl_len >= min_len,
                             infl_len <= max_len)

    if only_nonzero:
        # check to make sure the values of each contiguous block are True,
        # by checking the first value of each block
        infl_ok = np.logical_and(infl_ok,
                                 data[infl[:-1]])

    # inflate start/end indexes into full ranges of values
    blocks = [np.arange(infl[i], infl[i + 1])
              for i, ok in enumerate(infl_ok) if ok]
    return blocks


def merge_intervals(intervals):
    '''
    Given a list of intervals, merge overlapping ranges into a single list
    of non- overlapping ranges

    Parameters
    -----------
    intervals: (n,2) list of [start, end] values for ranges

    Returns
    -----------
    merged: (m,2) list of [start, end] values of ranges with no overlaps
    '''
    def merge_generator(intervals):
        intervals.sort(axis=1)
        intervals = intervals[np.lexsort(intervals.T[::-1])]
        low, high = intervals[0]
        for current in intervals[1:]:
            if current[0] <= high:
                high = max(high, current[1])
            else:
                yield [low, high]
                low, high = current
        yield [low, high]

    intervals = np.asanyarray(intervals)
    if not util.is_shape(intervals, (-1, 2)):
        raise ValueError('Intervals must be (n,2)!')

    merged = np.array([i for i in merge_generator(intervals)])
    return merged
