import numpy as np
import time

from .constants import log

from . import util
from . import convex
from . import nsphere
from . import geometry
from . import grouping
from . import triangles
from . import transformations

try:
    from scipy import spatial
    from scipy import optimize
except ImportError:
    log.warning('Scipy import failed!')


def oriented_bounds_2D(points, qhull_options='QbB'):
    """
    Find an oriented bounding box for an array of 2D points.

    Parameters
    ----------
    points : (n,2) float
      Points in 2D.

    Returns
    ----------
    transform : (3,3) float
      Homogenous 2D transformation matrix to move the
      input points so that the axis aligned bounding box
      is CENTERED AT THE ORIGIN.
    rectangle : (2,) float
       Size of extents once input points are transformed
       by transform
    """
    # make sure input is a numpy array
    points = np.asanyarray(points, dtype=np.float64)
    # create a convex hull object of our points
    # 'QbB' is a qhull option which has it scale the input to unit
    # box to avoid precision issues with very large/small meshes
    convex = spatial.ConvexHull(
        points, qhull_options=qhull_options)

    # (n,2,3) line segments
    hull_edges = convex.points[convex.simplices]
    # (n,2) points on the convex hull
    hull_points = convex.points[convex.vertices]

    # direction of the edges of the hull polygon
    edge_vectors = np.diff(hull_edges, axis=1).reshape((-1, 2))

    # unitize vectors
    edge_vectors /= np.linalg.norm(edge_vectors, axis=1).reshape((-1, 1))
    # create a set of perpendicular vectors
    perp_vectors = np.fliplr(edge_vectors) * [-1.0, 1.0]

    # find the projection of every hull point on every edge vector
    # this does create a potentially gigantic n^2 array in memory,
    # and there is the 'rotating calipers' algorithm which avoids this
    # however, we have reduced n with a convex hull and numpy dot products
    # are extremely fast so in practice this usually ends up being pretty
    # reasonable
    x = np.dot(edge_vectors, hull_points.T)
    y = np.dot(perp_vectors, hull_points.T)

    # reduce the projections to maximum and minimum per edge vector
    bounds = np.column_stack((x.min(axis=1),
                              y.min(axis=1),
                              x.max(axis=1),
                              y.max(axis=1)))

    # calculate the extents and area for each edge vector pair
    extents = np.diff(bounds.reshape((-1, 2, 2)),
                      axis=1).reshape((-1, 2))
    area = np.product(extents, axis=1)
    area_min = area.argmin()

    # (2,) float of smallest rectangle size
    rectangle = extents[area_min]

    # find the (3,3) homogenous transformation which moves the input
    # points to have a bounding box centered at the origin
    offset = -bounds[area_min][:2] - (rectangle * .5)
    theta = np.arctan2(*edge_vectors[area_min][::-1])
    transform = transformations.planar_matrix(offset,
                                              theta)

    # we would like to consistently return an OBB with
    # the largest dimension along the X axis rather than
    # the long axis being arbitrarily X or Y.
    if np.less(*rectangle):
        # a 90 degree rotation
        flip = transformations.planar_matrix(theta=np.pi / 2)
        # apply the rotation
        transform = np.dot(flip, transform)
        # switch X and Y in the OBB extents
        rectangle = np.roll(rectangle, 1)

    return transform, rectangle


def oriented_bounds(obj, angle_digits=1, ordered=True):
    """
    Find the oriented bounding box for a Trimesh

    Parameters
    ----------
    obj : trimesh.Trimesh, (n, 2) float, or (n, 3) float
       Mesh object or points in 2D or 3D space
    angle_digits : int
       How much angular precision do we want on our result.
       Even with less precision the returned extents will cover
       the mesh albeit with larger than minimal volume, and may
       experience substantial speedups.

    Returns
    ----------
    to_origin : (4,4) float
      Transformation matrix which will move the center of the
      bounding box of the input mesh to the origin.
    extents: (3,) float
      The extents of the mesh once transformed with to_origin
    """

    # extract a set of convex hull vertices and normals from the input
    # we bother to do this to avoid recomputing the full convex hull if
    # possible
    if hasattr(obj, 'convex_hull'):
        # if we have been passed a mesh, use its existing convex hull to pull from
        # cache rather than recomputing. This version of the cached convex hull has
        # normals pointing in arbitrary directions (straight from qhull)
        # using this avoids having to compute the expensive corrected normals
        # that mesh.convex_hull uses since normal directions don't matter here
        vertices = obj.convex_hull.vertices
        hull_normals = obj.convex_hull.face_normals
    elif util.is_sequence(obj):
        # we've been passed a list of points
        points = np.asanyarray(obj)
        if util.is_shape(points, (-1, 2)):
            return oriented_bounds_2D(points)
        elif util.is_shape(points, (-1, 3)):
            hull_obj = spatial.ConvexHull(points)
            vertices = hull_obj.points[hull_obj.vertices]
            hull_normals, valid = triangles.normals(
                hull_obj.points[hull_obj.simplices])
        else:
            raise ValueError('Points are not (n,3) or (n,2)!')
    else:
        raise ValueError(
            'Oriented bounds must be passed a mesh or a set of points!')

    # convert face normals to spherical coordinates on the upper hemisphere
    # the vector_hemisphere call effectivly merges negative but otherwise
    # identical vectors
    spherical_coords = util.vector_to_spherical(
        util.vector_hemisphere(hull_normals))
    # the unique_rows call on merge angles gets unique spherical directions to check
    # we get a substantial speedup in the transformation matrix creation
    # inside the loop by converting to angles ahead of time
    spherical_unique = grouping.unique_rows(spherical_coords,
                                            digits=angle_digits)[0]

    min_volume = np.inf
    tic = time.time()

    for spherical in spherical_coords[spherical_unique]:
        # a matrix which will rotate each hull normal to [0,0,1]
        to_2D = np.linalg.inv(transformations.spherical_matrix(*spherical))
        # apply the transform here
        projected = np.dot(to_2D, np.column_stack(
            (vertices, np.ones(len(vertices)))).T).T[:, :3]

        height = projected[:, 2].ptp()
        rotation_2D, box = oriented_bounds_2D(projected[:, :2])
        volume = np.product(box) * height
        if volume < min_volume:
            min_volume = volume
            min_extents = np.append(box, height)
            min_2D = to_2D.copy()
            rotation_2D[:2, 2] = 0.0
            rotation_Z = transformations.planar_matrix_to_3D(rotation_2D)

    # combine the 2D OBB transformation with the 2D projection transform
    to_origin = np.dot(rotation_Z, min_2D)

    # transform points using our matrix to find the translation for the
    # transform
    transformed = transformations.transform_points(vertices,
                                                   to_origin)
    box_center = (transformed.min(axis=0) + transformed.ptp(axis=0) * .5)
    to_origin[:3, 3] = -box_center

    # return ordered 3D extents
    if ordered:
        # sort the three extents
        order = min_extents.argsort()
        # generate a matrix which will flip transform
        # to match the new ordering
        flip = np.eye(4)
        flip[:3, :3] = -np.eye(3)[order]

        # make sure transform isn't mangling triangles
        # by reversing windings on triangles
        if np.isclose(np.trace(flip[:3, :3]), 0.0):
            flip[:3, :3] = np.dot(flip[:3, :3], -np.eye(3))

        # apply the flip to the OBB transform
        to_origin = np.dot(flip, to_origin)
        # apply the order to the extents
        min_extents = min_extents[order]

    log.debug('oriented_bounds checked %d vectors in %0.4fs',
              len(spherical_unique),
              time.time() - tic)

    return to_origin, min_extents


def minimum_cylinder(obj, sample_count=6, angle_tol=.001):
    """
    Find the approximate minimum volume cylinder which contains
    a mesh or a a list of points.

    Samples a hemisphere then uses scipy.optimize to pick the
    final orientation of the cylinder.

    A nice discussion about better ways to implement this is here:
    https://www.staff.uni-mainz.de/schoemer/publications/ALGO00.pdf


    Parameters
    ----------
    obj : trimesh.Trimesh, or (n, 3) float
      Mesh object or points in space
    sample_count : int
      How densely should we sample the hemisphere.
      Angular spacing is 180 degrees / this number

    Returns
    ----------
    result : dict
      With keys:
        'radius'    : float, radius of cylinder
        'height'    : float, height of cylinder
        'transform' : (4,4) float, transform from the origin
                      to centered cylinder
    """

    def volume_from_angles(spherical, return_data=False):
        """
        Takes spherical coordinates and calculates the volume
        of a cylinder along that vector

        Parameters
        ---------
        spherical : (2,) float
           Theta and phi
        return_data : bool
           Flag for returned

        Returns
        --------
        if return_data:
            transform ((4,4) float)
            radius (float)
            height (float)
        else:
            volume (float)
        """
        to_2D = transformations.spherical_matrix(*spherical,
                                                 axes='rxyz')
        projected = transformations.transform_points(hull,
                                                     matrix=to_2D)
        height = projected[:, 2].ptp()

        try:
            center_2D, radius = nsphere.minimum_nsphere(projected[:, :2])
        except BaseException:
            # in degenerate cases return as infinite volume
            return np.inf

        volume = np.pi * height * (radius ** 2)
        if return_data:
            center_3D = np.append(center_2D, projected[
                                  :, 2].min() + (height * .5))
            transform = np.dot(np.linalg.inv(to_2D),
                               transformations.translation_matrix(center_3D))
            return transform, radius, height
        return volume

    # We've been passed a mesh with radial symmetry
    # Use center mass and symmetry axis and go home early
    if hasattr(obj, 'symmetry') and obj.symmetry == 'radial':
        if obj.is_watertight:
            # set origin to center of mass
            origin = obj.center_mass
        else:
            # convex hull should be watertight
            origin = obj.convex_hull.center_mass

        # will align symmetry axis with Z and move origin to zero
        to_2D = geometry.plane_transform(
            origin=origin,
            normal=obj.symmetry_axis)

        on_plane = transformations.transform_points(
            obj.vertices, to_2D)

        # radius is maximum radius
        radius = (on_plane[:, :2] ** 2).sum(axis=1).max() ** .5
        # height is overall Z span
        height = on_plane[:, 2].ptp()
        # save to kwargs
        result = {'height': height,
                  'radius': radius,
                  'transform': np.linalg.inv(to_2D)}
        return result

    hull = convex.hull_points(obj)
    if not util.is_shape(hull, (-1, 3)):
        raise ValueError('Input must be reducable to 3D points!')

    # sample a hemisphere so local hill climbing can do its thing
    samples = util.grid_linspace([[0, 0], [np.pi, np.pi]], sample_count)

    # if it's rotationally symmetric the bounding cylinder
    # is almost certainly along one of the PCI vectors
    if hasattr(obj, 'principal_inertia_vectors'):
        # add the principal inertia vectors if we have a mesh
        samples = np.vstack(
            (samples,
             util.vector_to_spherical(obj.principal_inertia_vectors)))

    tic = [time.time()]
    # the projected volume at each sample
    volumes = np.array([volume_from_angles(i) for i in samples])
    # the best vector in (2,) spherical coordinates
    best = samples[volumes.argmin()]
    tic.append(time.time())

    # since we already explored the global space, set the bounds to be
    # just around the sample that had the lowest volume
    step = 2 * np.pi / sample_count
    bounds = [(best[0] - step, best[0] + step),
              (best[1] - step, best[1] + step)]
    # run the local optimization
    r = optimize.minimize(volume_from_angles,
                          best,
                          tol=angle_tol,
                          method='SLSQP',
                          bounds=bounds)

    tic.append(time.time())
    log.info('Performed search in %f and minimize in %f', *np.diff(tic))

    # actually chunk the information about the cylinder
    transform, radius, height = volume_from_angles(r['x'], return_data=True)
    result = {'transform': transform,
              'radius': radius,
              'height': height}
    return result


def corners(bounds):
    """
    Given a pair of axis aligned bounds, return all
    8 corners of the bounding box.

    Parameters
    ----------
    bounds : (2,3) or (2,2) float
      Axis aligned bounds

    Returns
    ----------
    corners : (8,3) float
      Corner vertices of the cube
    """

    bounds = np.asanyarray(bounds, dtype=np.float64)

    if util.is_shape(bounds, (2, 2)):
        bounds = np.column_stack((bounds, [0, 0]))
    elif not util.is_shape(bounds, (2, 3)):
        raise ValueError('bounds must be (2,2) or (2,3)!')

    minx, miny, minz, maxx, maxy, maxz = np.arange(6)
    corner_index = np.array([minx, miny, minz,
                             maxx, miny, minz,
                             maxx, maxy, minz,
                             minx, maxy, minz,
                             minx, miny, maxz,
                             maxx, miny, maxz,
                             maxx, maxy, maxz,
                             minx, maxy, maxz]).reshape((-1, 3))

    corners = bounds.reshape(-1)[corner_index]
    return corners


def contains(bounds, points):
    """
    Do an axis aligned bounding box check on a list of points.

    Parameters
    -----------
    bounds : (2, dimension) float
       Axis aligned bounding box
    points : (n, dimension) float
       Points in space

    Returns
    -----------
    points_inside : (n,) bool
      True if points are inside the AABB
    """
    # make sure we have correct input types
    bounds = np.asanyarray(bounds, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)

    if len(bounds) != 2:
        raise ValueError('bounds must be (2,dimension)!')
    if not util.is_shape(points, (-1, bounds.shape[1])):
        raise ValueError('bounds shape must match points!')

    # run the simple check
    points_inside = np.logical_and(
        (points > bounds[0]).all(axis=1),
        (points < bounds[1]).all(axis=1))

    return points_inside
