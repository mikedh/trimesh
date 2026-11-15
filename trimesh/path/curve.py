import numpy as np

from ..constants import res_path as res
from ..constants import tol_path as tol
from ..typed import Integer


def discretize_bezier(points, count=None, scale=1.0):
    """
    Parameters
    ----------
    points : (order, dimension) float
      Control points of the bezier curve
      For a 2D cubic bezier, order=3, dimension=2
    count : int, or None
      Number of segments
    scale : float
      Scale of curve
    Returns
    ----------
    discrete: (n, dimension) float
     Points forming a a polyline representation
    """
    # make sure we have a numpy array
    points = np.asanyarray(points, dtype=np.float64)

    if count is None:
        # how much distance does a small percentage of the curve take
        # this is so we can figure out how finely we have to sample t
        norm = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        count = np.ceil(norm / (res.seg_frac * scale))
        count = int(
            np.clip(count, res.min_sections * len(points), res.max_sections * len(points))
        )
    count = int(count)

    # parameterize incrementing 0.0 - 1.0
    t = np.linspace(0.0, 1.0, count)
    # decrementing 1.0-0.0
    t_d = 1.0 - t
    n = len(points) - 1
    # binomial coefficients, i, and each point
    iterable = zip(binomial(n), np.arange(len(points)), points)
    # run the actual interpolation
    stacked = [
        ((t**i) * (t_d ** (n - i))).reshape((-1, 1)) * p * c for c, i, p in iterable
    ]
    result = np.sum(stacked, axis=0)

    # a bezier curve always starts and ends on control points
    if tol.strict:
        # test to make sure end points are correct
        test = np.sum((result[[0, -1]] - points[[0, -1]]) ** 2, axis=1)
        assert (test < tol.merge).all()
        assert len(result) >= 2

    # snap the first and last points to the exact control point
    result[[0, -1]] = points[[0, -1]]

    return result


def bounds_bezier(points):
    """Conservatively bound a Bezier curve with subdivided control hulls."""
    points = np.asanyarray(points, dtype=np.float64)
    scale = np.abs(points).max(axis=0)
    tolerance = scale * 1e-12
    curves = points[None]
    endpoints = points[[0, -1]]
    lower, upper = endpoints.min(axis=0), endpoints.max(axis=0)
    bounds = np.array([lower, upper])
    transform = None
    # Each subcurve remains inside its control hull. Only refine hulls which
    # could extend the global bounds, rather than sampling the whole curve.
    for depth in range(27):
        lower = np.minimum(lower, curves[:, [0, -1]].min(axis=(0, 1)))
        upper = np.maximum(upper, curves[:, [0, -1]].max(axis=(0, 1)))
        low, high = curves.min(axis=1), curves.max(axis=1)
        done = ((low >= lower - tolerance) & (high <= upper + tolerance)).all(axis=1)
        if depth == 26:
            done[:] = True
        if done.any():
            bounds[0] = np.minimum(bounds[0], low[done].min(axis=0))
            bounds[1] = np.maximum(bounds[1], high[done].max(axis=0))
        curves = curves[~done]
        if len(curves) == 0:
            break
        if transform is None:
            # Four quarter-curve transforms, built once with de Casteljau.
            work = np.eye(len(points))
            left, right = [work[0]], [work[-1]]
            for _ in range(len(points) - 1):
                work = 0.5 * work[:-1] + 0.5 * work[1:]
                left.append(work[0])
                right.append(work[-1])
            halves = np.stack((left, right[::-1]))
            transform = np.matmul(halves[:, None], halves[None]).reshape(
                4, len(points), len(points)
            )
        curves = np.matmul(transform[:, None], curves[None]).reshape(
            -1, len(points), points.shape[1]
        )
    padding = 128 * np.finfo(np.float64).eps * scale * len(points) ** 2
    return bounds + np.array([-padding, padding])


def bounds_bspline(control, knots):
    """Bound each polynomial span, including both sides of repeated knots."""
    from scipy.interpolate import PPoly

    control = np.asanyarray(control, dtype=np.float64)
    degree = len(knots) - len(control) - 1
    bounds = []
    edges = np.unique(knots)
    for axis in control.T:
        polynomial = PPoly.from_spline((knots, axis, degree))
        # Match discretize_bspline's full knot interval, including extrapolation.
        roots = polynomial.derivative().roots(extrapolate=True)
        roots = roots[np.isfinite(roots) & (roots >= edges[0]) & (roots <= edges[-1])]
        parameters = np.concatenate((edges, np.nextafter(edges[1:], -np.inf), roots))
        values = polynomial(parameters)
        bounds.append([values.min(), values.max()])
    bounds = np.array(bounds).T
    scale = np.maximum(np.abs(control).max(axis=0), np.abs(bounds).max(axis=0))
    padding = 128 * np.finfo(np.float64).eps * scale * (degree + 1) ** 2
    return bounds + np.array([-padding, padding])


def discretize_bspline(control, knots, count=None, scale=1.0):
    """
    Given a B-Splines control points and knot vector, return
    a sampled version of the curve.

    Parameters
    ----------
    control : (o, d) float
      Control points of the b- spline
    knots : (j,) float
      B-spline knots
    count : int
      Number of line segments to discretize the spline
      If not specified will be calculated as something reasonable

    Returns
    ----------
    discrete : (count, dimension) float
       Points on a polyline version of the B-spline
    """

    # evaluate the b-spline using scipy/fitpack
    from scipy.interpolate import splev

    # (n, d) control points where d is the dimension of vertices
    control = np.asanyarray(control, dtype=np.float64)
    degree = len(knots) - len(control) - 1
    if count is None:
        norm = np.linalg.norm(np.diff(control, axis=0), axis=1).sum()
        count = int(
            np.clip(
                norm / (res.seg_frac * scale),
                res.min_sections * len(control),
                res.max_sections * len(control),
            )
        )

    ipl = np.linspace(knots[0], knots[-1], count)
    discrete = splev(ipl, [knots, control.T, degree])
    discrete = np.column_stack(discrete)

    return discrete


def binomial(n: Integer) -> list:
    """
    Return all binomial coefficients for a given order.

    For n > 5, scipy.special.binom is used, below we hardcode.

    Parameters
    --------------
    n : int
      Order of binomial

    Returns
    ---------------
    binom : (n + 1,) int
      Binomial coefficients of a given order
    """
    if n == 1:
        return [1, 1]
    elif n == 2:
        return [1, 2, 1]
    elif n == 3:
        return [1, 3, 3, 1]
    elif n == 4:
        return [1, 4, 6, 4, 1]
    elif n == 5:
        return [1, 5, 10, 10, 5, 1]
    else:
        from scipy.special import binom

        return binom(n, np.arange(n + 1))
