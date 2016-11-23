import numpy as np

from collections import deque

from . import arc
from . import entities

from ..nsphere import fit_nsphere
from ..util import unitize, diagonal_dot
from ..constants import log
from ..constants import tol_path as tol


def fit_circle_check(points, prior=None, scale=1.0, verbose=False):
    '''
    Fit a circle, and reject the fit if:
    * the radius is larger than tol.radius_min*scale or tol.radius_max*scale
    * any segment spans more than tol.seg_angle
    * any segment is longer than tol.seg_frac*scale
    * the fit deviates by more than tol.radius_frac*radius
    * the segments on the ends deviate from tangent by more than tol.tangent

    Arguments
    ---------
    points:  (n, d) set of points which represent a path
    prior:   (center, radius) tuple for best guess, or None if unknown
    scale:   float, what is the overall scale of the set of points
    verbose: boolean, if True output log.debug messages for the reasons
             for fit rejection. Potentially generates hundreds of thousands of
             messages so only suggested in manual debugging.

    Returns
    ---------
    if fit is acceptable:
        (center, radius) tuple
    else:
        None
    '''
    # an arc needs at least three points
    if len(points) < 3:
        return None

    # do a least squares fit on the points
    C, R, r_deviation = fit_nsphere(points, prior=prior)

    # check to make sure radius is between min and max allowed
    if not tol.radius_min < (R / scale) < tol.radius_max:
        if verbose:
            log.debug('circle fit error: R %f', R / scale)
        return None

    # check point radius error
    r_error = r_deviation / R
    if r_error > tol.radius_frac:
        if verbose:
            log.debug('circle fit error: fit %s', str(r_error))
        return None

    vectors = np.diff(points, axis=0)
    segment = np.linalg.norm(vectors, axis=1)

    # check segment length as a fraction of drawing scale
    scaled = segment / scale
    # approximate angle in radians, segments are linear length
    # not arc length but this is close and avoids a cosine
    angle = segment / R

    if (angle > tol.seg_angle).any():
        if verbose:
            log.debug('circle fit error: angle %s', str(angle))
        return None

    if (scaled > tol.seg_frac).any():
        if verbose:
            log.debug('circle fit error: segment %s', str(scaled))
        return None

    # check to make sure the line segments on the ends are actually
    # tangent with the candidate circle fit
    mid_pt = points[[0, -2]] + (vectors[[0, -1]] * .5)
    radial = unitize(mid_pt - C)
    ends = unitize(vectors[[0, -1]])
    tangent = np.abs(np.arccos(diagonal_dot(radial, ends)))
    tangent = np.abs(tangent - np.pi / 2).max()
    if tangent > tol.tangent:
        if verbose:
            log.debug('circle fit error: tangent %f',
                      np.degrees(tangent))
        return None

    return (C, R)


def is_circle(points, scale, verbose=True):
    '''
    Given a set of points, quickly determine if they represent
    a circle or not.
    '''

    # make sure input is a numpy array
    points = np.asanyarray(points)
    scale = float(scale)

    # can only be a circle if the first and last point are the
    # same (AKA is a closed path)
    if np.linalg.norm(points[0] - points[-1]) > tol.merge:
        return None

    box = points.ptp(axis=0)
    # the bounding box size of the points
    # check aspect ratio as an early exit if the path is not a circle
    aspect = np.divide(*box)
    if np.abs(aspect - 1.0) > tol.aspect_frac:
        return None

    # fit a circle with tolerance checks
    CR = fit_circle_check(points, scale=scale)
    if CR is None:
        return None

    # return the circle as three control points
    control = arc.angles_to_threepoint([0, np.pi * .5], *CR)
    return control


def arc_march(points, scale):
    '''
    Split a path into line and arc segments, using least squares fit.

    Arguments
    ---------
    points: (n,d) points

    Returns:
    arcs:  (b) sequence of points indices that could be replaced with an arc
    '''

    def finalize_arc(points_id):
        # do final checks on the points contained in current and append them
        # to the list of arcs if they pass
        points_id = np.array(points_id)
        points_id = points_id[[0, int(len(points_id) / 2.0), -1]]

        try:
            center_info = arc.arc_center(points[points_id])
            C, R, N, A = (center_info['center'],
                          center_info['radius'],
                          center_info['normal'],
                          center_info['span'])
        except ValueError:
            log.warning('Skipping candidate arc!', exc_info=True)
            return

        span = scale * (A / R)
        if span > 1.5:
            arcs.append(points_id)
        else:
            log.debug('Arc failed span test: %f', span)

    points = np.asanyarray(points)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge
    count = len(points)
    scale = float(scale)
    # if scale is None:
    #    scale = np.ptp(points, axis=0).max()

    arcs = deque()
    current = deque()
    prior = None

    # how many times to go through points
    # if the points are closed go through them up to twice
    attempts = count + count * int(closed)

    for index in range(attempts):
        # have we already traversed these points
        looped = index >= count
        # make sure we stay in range
        i = index % count

        # if we looped over, it means that these points are closed
        # and if they are closed it means points[0] == points[-1]
        # thus, once we go over them the second time we want to skip index
        # zero to avoid processing a duplicate point.
        if looped and i == 0:
            continue

        # add an index to the current set of candidates
        current.append(i)
        # if the current number of candidates is less than three they can't be
        # an arc
        if (len(current) < 3):
            continue

        # fit a circle to the points, and reject the fit if tolerances aren't met
        # if the fit is rejected, fit_circle_check will return None
        checked = fit_circle_check(points[current],
                                   prior=prior,
                                   scale=scale,
                                   verbose=True)
        arc_ok = checked is not None

        # since we are going over the points twice, on the second pass we only
        # want to go until an arc fit fails
        ending = looped and (not arc_ok)
        ending = ending or (index >= attempts - 1)

        if ending and prior is not None:
            # we could be stopping for a bad fit,
            # or we could have just ran out of indexes.
            # if its a bad fit, remove the last point added.
            if not arc_ok:
                current.pop()
            # if we are stopping and have gotten an acceptable fit add the arc
            finalize_arc(current)
        elif arc_ok:
            # if we aren't ending and the fit looks good
            # just update the prior with the fit
            prior = checked[0]
        elif prior is None:
            # we haven't seen an acceptable fit
            # so remove an index from the left
            current.popleft()
        else:
            # the arc isn't ok, and we have a fit so remove
            # the latest point then add the arc
            current.pop()
            finalize_arc(current)
            # reset the candidates
            current = deque([i - 1, i])
            prior = None

        if ending:
            break

    if looped and len(arcs) > 0 and arcs[0][0] == 0:
        arcs.popleft()

    arcs = np.array(arcs)
    return arcs


def merge_colinear(points, scale=None):
    '''
    Given a set of points representing a path in space,
    merge points which are colinear.

    Arguments
    ----------
    points: (n, d) set of points (where d is dimension)
    scale:  float, scale of drawing
    Returns
    ----------
    merged: (j, d) set of points with colinear and duplicate
             points merged, where (j < n)
    '''
    points = np.array(points)
    if scale is None:
        scale = np.ptp(points, axis=0).max()

    # the vector from one point to the next
    direction = np.diff(points, axis=0)
    # the length of the direction vector
    direction_norm = np.linalg.norm(direction, axis=1)
    # make sure points don't have zero length
    direction_ok = direction_norm > tol.merge

    # remove duplicate points
    points = np.vstack((points[0], points[1:][direction_ok]))
    direction = direction[direction_ok]
    direction_norm = direction_norm[direction_ok]

    # change nonzero direction vectors to unit vectors
    direction /= direction_norm.reshape((-1, 1))
    # find the difference between subsequent direction vectors
    direction_diff = np.linalg.norm(np.diff(direction, axis=0), axis=1)

    # magnitude of direction difference between vectors times direction length
    colinear = (direction_diff * direction_norm[1:]) < (tol.merge * scale)
    colinear_index = np.nonzero(colinear)[0]

    mask = np.ones(len(points), dtype=np.bool)
    # since we took diff, we need to offset by one
    mask[colinear_index + 1] = False
    merged = points[mask]
    return merged


def resample_spline(points, smooth=.001, count=None, degree=3):
    from scipy.interpolate import splprep, splev
    if count is None:
        count = len(points)
    points = np.asanyarray(points)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge

    tpl = splprep(points.T, s=smooth, k=degree)[0]
    i = np.linspace(0.0, 1.0, count)
    resampled = np.column_stack(splev(i, tpl))

    if closed:
        shared = resampled[[0, -1]].mean(axis=0)
        resampled[0] = shared
        resampled[-1] = shared

    return resampled


def points_to_spline_entity(points, smooth=.0005, count=None):
    from scipy.interpolate import splprep

    if count is None:
        count = len(points)
    points = np.asanyarray(points)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge

    knots, control, degree = splprep(points.T, s=smooth)[0]
    control = np.transpose(control)
    index = np.arange(len(control))

    if closed:
        control[0] = control[[0, -1]].mean(axis=0)
        control = control[:-1]
        index[-1] = index[0]

    entity = entities.BSpline(points=index,
                              knots=knots,
                              closed=closed)

    return entity, control


def three_point(indices):
    result = [indices[0],
              indices[int(len(indices) / 2)],
              indices[-1]]
    return np.array(result)


def polygon_to_cleaned(polygon, scale):
    buffered = polygon.buffer(0.0)
    points = merge_colinear(buffered.exterior.coords)
    return points


def simplify_path(drawing):
    '''
    Simplify a path containing only line sections into one with fit arcs and circles.
    '''

    if any([i.__class__.__name__ != 'Line' for i in drawing.entities]):
        log.debug('Path contains non- linear entities, skipping')
        return

    vertices_new = deque()
    entities_new = deque()

    for path_index in range(len(drawing.paths)):
        points = polygon_to_cleaned(drawing.polygons_closed[path_index],
                                    scale=drawing.scale)
        circle = is_circle(points, scale=drawing.scale)

        if circle is not None:
            entities_new.append(entities.Arc(points=np.arange(3) + len(vertices_new),
                                             closed=True))
            vertices_new.extend(circle)
        else:
            arc_idx = arc_march(points, scale=drawing.scale)
            if len(arc_idx) > 0:
                for arc in arc_idx:
                    entities_new.append(entities.Arc(points=three_point(arc) + len(vertices_new),
                                                     closed=False))
                line_idx = infill_lines(
                    arc_idx, len(points)) + len(vertices_new)
            else:
                line_idx = pair_space(0, len(points) - 1) + len(vertices_new)
                line_idx = [np.mod(np.arange(len(points) + 1),
                                   len(points)) + len(vertices_new)]

            for line in line_idx:
                entities_new.append(entities.Line(points=line))
            vertices_new.extend(points)

    drawing._cache.clear()
    drawing.vertices = np.array(vertices_new)
    drawing.entities = np.array(entities_new)


def pair_space(start, end):
    if start == end:
        return []
    idx = np.arange(start, end + 1)
    idx = np.column_stack((idx, idx)).reshape(-1)[1:-1].reshape(-1, 2)
    return idx


def infill_lines(idxs, idx_max):
    if len(idxs) == 0:
        return np.array([])
    ends = np.array([i[[0, -1]] for i in idxs])
    ends = np.roll(ends.reshape(-1), -1).reshape(-1, 2)

    if np.greater(*ends[-1]):
        ends[-1][1] += idx_max

    infill = np.diff(ends, axis=1).reshape(-1) > 0
    aranges = ends[infill]
    if len(aranges) == 0:
        return np.array([])
    result = np.vstack([pair_space(*i) for i in aranges])
    result %= idx_max
    return result
