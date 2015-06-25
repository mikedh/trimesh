import numpy as np
import logging

from .constants import TOL_ZERO

def unitize(points, check_valid=False):
    '''
    Flexibly turn a list of vectors into a list of unit vectors.
    
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
    points       = np.array(points)
    axis         = len(points.shape) - 1
    length       = np.sum(points ** 2, axis=axis) ** .5
    if check_valid:
        valid = np.greater(length, TOL_ZERO)
        if axis == 1: 
            unit_vectors = (points[valid].T / length[valid]).T
        elif len(points.shape) == 1 and valid: 
            unit_vectors = points / length
        else:         
            unit_vectors = []
        return unit_vectors, valid        
    else: 
        unit_vectors = (points.T / length).T
    return unit_vectors

def transformation_2D(offset=[0.0,0.0], theta=0.0):
    '''
    2D homogeonous transformation matrix
    '''
    T = np.eye(3)
    s = np.sin(theta)
    c = np.cos(theta)

    T[0,0:2] = [ c, s]
    T[1,0:2] = [-s, c]
    T[0:2,2] = offset
    return T

def euclidean(a, b):
    '''
    Euclidean distance between vectors a and b
    '''
    return np.sum((np.array(a) - b)**2) ** .5

def is_sequence(arg):
    '''
    Returns true if arg is a sequence
    '''
    seq = (not hasattr(arg, "strip") and
           hasattr(arg, "__getitem__") or
           hasattr(arg, "__iter__"))

    # numpy sometimes returns objects that are single float64 values
    # but sure look like sequences, so we check the shape
    if hasattr(arg, 'shape'):
        seq = seq and arg.shape != ()

    return seq

def is_ccw(points):
    '''
    Given an (n,2) set of points, return True if they are counterclockwise
    '''
    xd = np.diff(points[:,0])
    yd = np.sum(np.column_stack((points[:,1], 
                                 points[:,1])).reshape(-1)[1:-1].reshape((-1,2)), axis=1)
    area = np.sum(xd*yd)*.5
    return area < 0

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
    points = np.array(points)
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
    x_grid = np.arange(*bounds[:,0], step = step)
    y_grid = np.arange(*bounds[:,1], step = step)
    grid   = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1,2))
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
    x_grid = np.linspace(*bounds[:,0], count = count)
    y_grid = np.linspace(*bounds[:,1], count = count)
    grid   = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1,2))
    return grid

def replace_references(data, reference_dict):
    # Replace references in place
    view = np.array(data).view().reshape((-1))
    for i, value in enumerate(view):
        if value in reference_dict:
            view[i] = reference_dict[value]
    return view

def attach_to_log(log_level=logging.DEBUG, blacklist=[]):
    '''
    Attach a stream handler to all loggers, so their output can be seen
    on the console
    '''
    try: 
        from colorlog import ColoredFormatter
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(filename)17s:%(lineno)-4s  %(blue)4s%(message)s",
            datefmt = None,
            reset   = True,
            log_colors = {'DEBUG':    'cyan',
                          'INFO':     'green',
                          'WARNING':  'yellow',
                          'ERROR':    'red',
                          'CRITICAL': 'red' } )
    except ImportError: 
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", 
            "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(log_level)

    for logger in logging.Logger.manager.loggerDict.values():
        if logger.__class__.__name__ != 'Logger': continue
        if (logger.name in ['TerminalIPythonApp',
                            'PYREADLINE'] or 
            logger.name in blacklist):
            continue
        logger.addHandler(handler_stream)
        logger.setLevel(log_level)
    np.set_printoptions(precision=4, suppress=True)
