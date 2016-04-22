import numpy as np
import time

from .util       import unitize, transformation_2D
from .constants  import log
from .grouping   import group_vectors
from .points     import transform_points, project_to_plane
from .geometry   import rotation_2D_to_3D

from scipy.spatial import ConvexHull

def oriented_bounds_2D(points):
    c = ConvexHull(np.asanyarray(points))
    hull = c.points[c.simplices]
    edge_vectors = unitize(np.diff(hull, axis=1).reshape((-1,2)))
    perp_vectors = np.fliplr(edge_vectors) * [-1.0,1.0]
    bounds = np.zeros((len(edge_vectors), 4))
    dt = hull.reshape((-1,2)).T
    for i, edge, perp in zip(range(len(edge_vectors)),
                             edge_vectors, 
                             perp_vectors):
        a = np.dot(edge, dt)
        b = np.dot(perp, dt)
        bounds[i] = [a.min(), b.min(), a.max(), b.max()]    
    extents = np.diff(bounds.reshape((-1,2,2)), axis=1).reshape((-1,2))
    area = np.product(extents, axis=1)
    area_min = area.argmin()
    offset = -bounds[area_min][0:2]
    theta = np.arctan2(*edge_vectors[area_min][::-1])
    transform = transformation_2D(offset, theta)
    rectangle = extents[area_min]

    return rectangle, transform

def oriented_bounds(mesh, angle_tol=1e-6):
    hull = mesh.convex_hull
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
        box, rotation_2D = oriented_bounds_2D(projected[:,0:2])
        volume = np.product(box) * height
        toc = time.time()
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
