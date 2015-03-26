import numpy as np
from collections import deque

from .geometry import unitize
from .constants import *

def merge_vertices_hash(mesh):
    '''
    Removes duplicate vertices, based on integer hashes.
    This is roughly 20x faster than querying a KD tree in a loop
    '''
    pre_merge = len(mesh.vertices)

    unique, inverse = unique_rows(mesh.vertices, return_inverse=True)        
    mesh.update_vertices(unique, inverse)
    log.debug('merge_vertices_hash reduced vertex count from %i to %i.',
              pre_merge,
              len(mesh.vertices))

def merge_vertices_kdtree(mesh, max_angle=None):
    '''
    Merges vertices which are identical, AKA within 
    Cartesian distance TOL_MERGE of each other.  
    Then replaces references in mesh.faces
    
    If max_angle == None, vertex normals won't be looked at. 
    if max_angle has a value, vertices will only be considered identical
    if they are within TOL_MERGE of each other, and the angle between
    their normals is less than angle_max

    Performance note:
    cKDTree requires scipy >= .12 for this query type and you 
    probably don't want to use plain python KDTree as it is crazy slow (~1000x in tests)
    '''
    from scipy.spatial import cKDTree as KDTree
    
    tree    = KDTree(mesh.vertices)
    used    = np.zeros(len(mesh.vertices), dtype=np.bool)
    inverse = np.arange(len(mesh.vertices), dtype=np.int)
    unique  = deque()
    
    if max_angle != None: mesh.verify_normals()

    for index, vertex in enumerate(mesh.vertices):
        if used[index]: continue
        neighbors = np.array(tree.query_ball_point(mesh.vertices[index], TOL_MERGE))
        used[[neighbors]] = True
        if max_angle != None:
            normals, aligned = group_vectors(mesh.vertex_normals[[neighbors]], 
                                             max_angle = max_angle)
            for group in aligned:
                inverse[neighbors[[group]]] = len(unique)
                unique.append(neighbors[group[0]])
        else:
            inverse[neighbors] = neighbors[0]
            unique.append(neighbors[0])

    mesh.update_vertices(unique, inverse)
   
    log.debug('merge_vertices_kdtree reduced vertex count from %i to %i', 
              len(used),
              len(unique))

def replace_references(data, reference_dict):
    '''
    Replace elements in an array as per a dictionary of replacement values. 

    Arguments
    ----------
    data:           numpy array 
    reference_dict: dictionary of replacement value mapping, eg: {2:1, 3:1, 4:5}
    '''
    shape = np.shape(data)
    view  = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view.reshape(shape)

def group(values, min_length=0, max_length=np.inf):
    '''
    Return the indices of values that are identical
    
    Arguments
    ----------
    values:     1D array 
    min_length: int, the shortest group allowed
                All groups will have len >= min_length
    max_length: int, the longest group allowed
                All groups will have len <= max_length
    
    Returns
    ----------
    groups: sequence of indices to form groups
            IE [0,1,0,1] returns [[0,2], [1,3]]
    '''
    order     = values.argsort()
    values    = values[order]
    dupe      = np.greater(np.abs(np.diff(values)), TOL_ZERO)
    dupe_idx  = np.append(0, np.nonzero(dupe)[0] + 1)
    dupe_len  = np.diff(np.hstack((dupe_idx, len(values)))) 
    dupe_ok   = np.logical_and(np.greater_equal(dupe_len, min_length),
                               np.less_equal(   dupe_len, max_length))
    groups    = [order[i:(i+j)] for i, j in zip(dupe_idx[dupe_ok], dupe_len[dupe_ok])]
    return groups
    
def hashable_rows(data, digits=None):
    '''
    We turn our array into integers, based on the precision 
    given by digits, and then put them in a hashable format. 
    
    Arguments
    ---------
    data:    (n,m) input array
    digits:  how many digits to add to hash, if data is floating point
             If none, TOL_MERGE will be turned into a digit count and used. 
    
    Returns
    ---------
    hashable:  (n) length array of custom data which can be sorted 
                or used as hash keys
    '''
    data = np.array(data)   
    if digits == None: digits = abs(int(np.log10(TOL_MERGE)))
     
    if data.dtype.kind in 'ib':
        #if data is an integer or boolean, don't bother multiplying by precision
        as_int = data
    else:
        as_int = ((data+10**-(digits+1))*10**digits).astype(np.int64)    
    hashable = np.ascontiguousarray(as_int).view(np.dtype((np.void, 
                                                         as_int.dtype.itemsize * as_int.shape[1]))).reshape(-1)
    return hashable
    
def unique_rows(data, return_inverse=False, digits=None):
    '''
    Returns indices of unique rows. It will return the 
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]
    '''
    hashes                   = hashable_rows(data, digits=digits)
    garbage, unique, inverse = np.unique(hashes, 
                                         return_index   = True, 
                                         return_inverse = True)
    if return_inverse: 
        return unique, inverse
    return unique
    
def group_rows(data, require_count=None, digits=None):
    '''
    Returns index groups of duplicate rows, for example:
    [[1,2], [3,4], [1,2]] will return [[0,2], [1]]
    
    Arguments
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
                   If require_count == None, shape will be irregular (AKA a sequence)
    '''
    
    def group_dict():
        '''
        Simple hash table based grouping. 
        The loop and appends make this rather slow on very large arrays,
        But it works on irregular groups nicely, unlike the slicing version of this function
        '''
        observed = dict()
        hashable = hashable_rows(data, digits=digits)
        for index, key in enumerate(hashable):
            key_string = key.tostring()
            if key_string in observed: observed[key_string].append(index)
            else:                      observed[key_string] = [index]
        return np.array(list(observed.values()))
        
    def group_slice():
        # create a representation of the rows that can be sorted
        hashable = hashable_rows(data, digits=digits)
        # record the order of the rows so we can get the original indices back later
        order    = np.argsort(hashable)
        # but for now, we want our hashes sorted
        hashable = hashable[order]
        # this is checking each neighbour for equality, example: 
        # example: hashable = [1, 1, 1]; dupe = [0, 0]
        dupe     = hashable[1:] != hashable[:-1]
        # we want the first index of a group, so we can slice from that location
        # example: hashable = [0 1 1]; dupe = [1,0]; dupe_idx = [0,1]
        dupe_idx = np.append(0, np.nonzero(dupe)[0] + 1)
        # if you wanted to use this one function to deal with non- regular groups
        # you could use: np.array_split(dupe_idx)
        # this is roughly 3x slower than using the group_dict method above. 
        start_ok   = np.diff(np.hstack((dupe_idx, len(hashable)))) == require_count
        groups     = np.tile(dupe_idx[start_ok].reshape((-1,1)), 
                             require_count) + np.arange(require_count)
        groups_idx = order[groups]
        if require_count == 1: 
            return groups_idx.reshape(-1)
        return groups_idx

    if require_count == None: return group_dict()
    else:                     return group_slice()

def group_vectors(vectors, 
                  max_angle        = np.radians(10), 
                  include_negative = False):
    '''
    Group vectors based on an angle tolerance, with the option to 
    include negative vectors. 
    
    This is very similar to a group_rows(stack_negative(rows))
    The main difference is that max_angle can be much looser, as we
    are doing actual distance queries. 
    '''
    from scipy.spatial import cKDTree as KDTree
    dist_max            = np.tan(max_angle)
    unit_vectors, valid = unitize(vectors, check_valid = True)
    valid_index         = np.nonzero(valid)[0]
    consumed            = np.zeros(len(unit_vectors), dtype=np.bool)
    tree                = KDTree(unit_vectors)
    unique_vectors      = deque()
    aligned_index       = deque()
    
    for index, vector in enumerate(unit_vectors):
        if consumed[index]: continue
        aligned = np.array(tree.query_ball_point(vector, dist_max))        
        if include_negative:
            aligned = np.append(aligned, tree.query_ball_point(-1*vector, dist_max))
        aligned = aligned.astype(int)
        consumed[[aligned]] = True
        unique_vectors.append(unit_vectors[aligned[-1]])
        aligned_index.append(valid_index[[aligned]])
    return np.array(unique_vectors), np.array(aligned_index)
    
def stack_negative(rows):
    '''
    Given an input of rows (n,d), return an array which is (n,2*d)
    Which is sign- independent
    '''
    rows     = np.array(rows)
    width    = rows.shape[1]
    stacked  = np.column_stack((rows, rows*-1))
    negative = rows[:,0] < 0
    stacked[negative] = np.roll(stacked[negative], 3, axis=1)
    return stacked
