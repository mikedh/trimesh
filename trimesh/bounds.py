import numpy as np
import time

from .constants import log

from .nsphere import minimum_nsphere

from . import util
from . import convex
from . import triangles
from . import grouping
from . import transformations

try:
    from scipy import spatial
    from scipy import optimize
except ImportError:
    log.warning('Scipy import failed!')


def oriented_bounds_2D(points):
    '''
    Find an oriented bounding box for a set of 2D points.

    Arguments
    ----------
    points: (n,2) float, 2D points

    Returns
    ----------
    transform: (3,3) float, homogenous 2D transformation matrix to move the input set of
               points so that the axis aligned bounding box is CENTERED AT THE ORIGIN
    rectangle: (2,) float, size of extents once input points are transformed by transform
    '''
    # make sure input is a numpy array
    points = np.asanyarray(points)
    # create a convex hull object of our points
    # 'QbB' is a qhull option which has it scale the input to unit box
    # to avoid precision issues with very large/small meshes
    convex = spatial.ConvexHull(points, qhull_options='QbB')

    # (n,2,3) line segments
    hull_edges = convex.points[convex.simplices]
    # (n,2) points on the convex hull
    hull_points = convex.points[convex.vertices]

    # unitize the direction of the edges of the hull polygon
    edge_vectors = util.unitize(np.diff(hull_edges, axis=1).reshape((-1, 2)))
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

    # calculate the extents and area that a box drawn around each edge_vector
    extents = np.diff(bounds.reshape((-1, 2, 2)), axis=1).reshape((-1, 2))
    area = np.product(extents, axis=1)
    area_min = area.argmin()

    #(2,) float of smallest rectangle size
    rectangle = extents[area_min]

    # find the (3,3) homogenous transformation which moves the input points
    # to have a bounding box centered at the origin
    offset = -bounds[area_min][0:2] - (rectangle * .5)
    theta = np.arctan2(*edge_vectors[area_min][::-1])
    transform = transformations.planar_matrix(offset, theta)

    return transform, rectangle


def oriented_bounds(obj, angle_digits=2):
    '''
    Find the oriented bounding box for a Trimesh

    Arguments
    ----------
    obj:       Trimesh object, (n,3) or (n,2) float set of points
    angle_tol: float, angle in radians that OBB can be away from minimum volume
               solution. Even with large values the returned extents will cover
               the mesh albeit with larger than minimal volume.
               Larger values may experience substantial speedups.
               Acceptable values are floats >= 0.0.
               The default is small (1e-6) but non-zero.

    Returns
    ----------
    to_origin: (4,4) float, transformation matrix which will move the center of the
               bounding box of the input mesh to the origin.
    extents: (3,) float, the extents of the mesh once transformed with to_origin
    '''

    # extract a set of convex hull vertices and normals from the input
    # we bother to do this to avoid recomputing the full convex hull if
    # possible
    if hasattr(obj, 'convex_hull_raw'):
        # if we have been passed a mesh, use its existing convex hull to pull from
        # cache rather than recomputing. This version of the cached convex hull has
        # normals pointing in arbitrary directions (straight from qhull)
        # using this avoids having to compute the expensive corrected normals
        # that mesh.convex_hull uses since normal directions don't matter here
        vertices = obj.convex_hull_raw.vertices
        hull_normals = obj.convex_hull_raw.face_normals
    elif util.is_sequence(obj):
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
    spherical_unique = grouping.unique_rows(
        spherical_coords, digits=angle_digits)[0]

    min_volume = np.inf
    tic = time.time()

    for spherical in spherical_coords[spherical_unique]:
        # a matrix which will rotate each hull normal to [0,0,1]
        to_2D = np.linalg.inv(transformations.spherical_matrix(*spherical))
        projected = transformations.transform_points(vertices, to_2D)

        height = projected[:, 2].ptp()
        rotation_2D, box = oriented_bounds_2D(projected[:, 0:2])
        volume = np.product(box) * height
        if volume < min_volume:
            min_volume = volume
            min_extents = np.append(box, height)
            min_2D = to_2D.copy()
            rotation_2D[0:2, 2] = 0.0
            rotation_Z = transformations.planar_matrix_to_3D(rotation_2D)

    # combine the 2D OBB transformation with the 2D projection transform
    to_origin = np.dot(rotation_Z, min_2D)

    # transform points using our matrix to find the translation for the
    # transform
    transformed = transformations.transform_points(vertices, to_origin)
    box_center = (transformed.min(axis=0) + transformed.ptp(axis=0) * .5)
    to_origin[0:3, 3] = -box_center

    log.debug('oriented_bounds checked %d vectors in %0.4fs',
              len(spherical_unique),
              time.time() - tic)

    return to_origin, min_extents


def minimum_cylinder(obj, sample_count=15, angle_tol=.001):
    '''
    Find the approximate minimum volume cylinder which contains a mesh or set of points.

    Samples a hemisphere in 12 degree increments and then uses scipy.optimize
    to pick the final orientation of the cylinder.

    A nice discussion about better ways to implement this is here:
    https://www.staff.uni-mainz.de/schoemer/publications/ALGO00.pdf


    Arguments
    ----------
    obj: Trimesh object OR
         (n,3) float, points in space
    sample_count: int, how densely should we sample the hemisphere.
                  Angular spacing is 180 degrees / this number

    Returns
    ----------
    result: dict, with keys:
                'radius'    : float, radius of cylinder
                'height'    : float, height of cylinder
                'transform' : (4,4) float, transform from the origin to centered cylinder
    '''

    def volume_from_angles(spherical, return_data=False):
        '''
        Takes spherical coordinates and calculates the volume of a cylinder
        along that vector

        Arguments
        ---------
        spherical: (2,) float, theta and phi
        return_data: bool, flag for returned

        Returns
        --------
        if return_data:
            transform ((4,4) float)
            radius (float)
            height (float)
        else:
            volume (float)
        '''
        to_2D = transformations.spherical_matrix(*spherical)
        projected = transformations.transform_points(hull, to_2D)
        height = projected[:, 2].ptp()
        # in degenerate cases return as infinite volume
        try:
            center_2D, radius = minimum_nsphere(projected[:, 0:2])
        except:
            return np.inf

        volume = np.pi * height * (radius ** 2)
        if return_data:
            center_3D = np.append(center_2D, projected[
                                  :, 2].min() + (height * .5))
            transform = np.dot(np.linalg.inv(to_2D),
                               transformations.translation_matrix(center_3D))
            return transform, radius, height
        return volume

    hull = convex.hull_points(obj)
    if not util.is_shape(hull, (-1, 3)):
        raise ValueError('Input must be reducable to 3D points!')

    # sample a hemisphere so local hill climbing can do its thing
    samples = util.grid_linspace_2D([[0, 0], [np.pi, np.pi]], sample_count)
    tic = [time.time()]
    # the best vector in (2,) spherical coordinates
    best = samples[np.argmin([volume_from_angles(i) for i in samples])]
    tic.append(time.time())

    # since we already explored the global space, set the bounds to be
    # just around the sample that had the lowest volume
    step = 2 * np.pi / sample_count
    bounds = [(best[0] - step, best[0] + step),
              (best[1] - step, best[1] + step)]

    # run the optimization
    r = optimize.minimize(volume_from_angles, best,
                          tol=angle_tol, method='SLSQP', bounds=bounds)
    tic.append(time.time())

    log.info('Performed search in %f and minimize in %f', *np.diff(tic))

    # actually chunk the information about the cylinder
    transform, radius, height = volume_from_angles(r['x'], return_data=True)
    result = {'transform': transform,
              'radius': radius,
              'height': height}
    return result
