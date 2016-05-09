import numpy as np
import time

from .util       import unitize, transformation_2D
from .constants  import log
from .grouping   import group_vectors
from .points     import transform_points, project_to_plane
from .geometry   import rotation_2D_to_3D

from scipy.spatial import ConvexHull

def oriented_bounds_2D(points):
    '''
    Find an oriented bounding box for a set of 2D points.

    Arguments
    ----------
    points: (n,2) float, 2D points
    
    Returns
    ----------
    transform: (3,3) float, homogenous 2D transformation matrix to move the input set of 
               points to the FIRST QUADRANT, so no value is negative. 
    rectangle: (2,) float, size of extents once input points are transformed by transform
    '''
    c = ConvexHull(np.asanyarray(points))
    # (n,2,3) line segments
    hull = c.points[c.simplices] 
    # (3,n) points on the hull to check against
    dot_test = c.points[c.vertices].reshape((-1,2)).T
    edge_vectors = unitize(np.diff(hull, axis=1).reshape((-1,2)))
    perp_vectors = np.fliplr(edge_vectors) * [-1.0,1.0]
    bounds = np.zeros((len(edge_vectors), 4))
    for i, edge, perp in zip(range(len(edge_vectors)),
                             edge_vectors, 
                             perp_vectors):
        x = np.dot(edge, dot_test)
        y = np.dot(perp, dot_test)
        bounds[i] = [x.min(), y.min(), x.max(), y.max()]

    extents  = np.diff(bounds.reshape((-1,2,2)), axis=1).reshape((-1,2))
    area     = np.product(extents, axis=1)
    area_min = area.argmin()

    offset = -bounds[area_min][0:2]
    theta  = np.arctan2(*edge_vectors[area_min][::-1])

    transform = transformation_2D(offset, theta)
    rectangle = extents[area_min]

    return transform, rectangle

def oriented_bounds(mesh, angle_tol=1e-6):
    '''
    Find the oriented bounding box for a Trimesh 

    Arguments
    ----------
    mesh: Trimesh object
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
    # this version of the cached convex hull has normals pointing in 
    # arbitrary directions (straight from qhull)
    # using this avoids having to compute the expensive corrected normals
    # that mesh.convex_hull uses since normal directions don't matter here
    hull = mesh._convex_hull_raw
    vectors = group_vectors(hull.face_normals, 
                            angle=angle_tol,
                            include_negative=True)[0]
    min_volume = np.inf
    tic = time.time()
    for i, normal in enumerate(vectors):
        projected, to_3D = project_to_plane(hull.vertices,
                                            plane_normal     = normal,
                                            return_planar    = False,
                                            return_transform = True)
        height = projected[:,2].ptp()
        rotation_2D, box = oriented_bounds_2D(projected[:,0:2])
        volume = np.product(box) * height
        if volume < min_volume:
            min_volume = volume
            rotation_2D[0:2,2] = 0.0
            rotation_Z = rotation_2D_to_3D(rotation_2D)
            to_2D = np.linalg.inv(to_3D)
            extents = np.append(box, height)
    to_origin = np.dot(rotation_Z, to_2D)
    transformed = transform_points(hull.vertices, to_origin)
    box_center = (transformed.min(axis=0) + transformed.ptp(axis=0)*.5)
    to_origin[0:3, 3] = -box_center
 
    log.debug('oriented_bounds checked %d vectors in %0.4fs',
              len(vectors),
              time.time()-tic)
    return to_origin, extents
