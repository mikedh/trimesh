import numpy as np
import time

from shapely.ops import polygonize

from .constants     import log
from .convex        import planar_hull
from .path.polygons import polygon_obb
from .grouping      import group_vectors
from .points        import transform_points
from .geometry      import rotation_2D_to_3D

def oriented_bounds(mesh, angle_tol=1e-6):
    hull = mesh.convex_hull
    vectors = group_vectors(hull.face_normals, 
                            angle=angle_tol,
                            include_negative=True)[0]
    min_volume = np.inf
    for i, normal in enumerate(vectors):
        tic = time.time()
        lines, to_3D, height = planar_hull(hull.vertices, normal)
        
        polygon = polygonize(lines).next()
        box, rotation_2D = polygon_obb(polygon)
        volume = np.product(box) * np.ptp(height)
        toc = time.time()
        if volume < min_volume:
            min_volume = volume
            rotation_2D[0:2,2] = 0.0
            rotation_Z = rotation_2D_to_3D(rotation_2D)
            to_2D = np.linalg.inv(to_3D)
            extents = np.append(box, np.ptp(height))
            log.info('Found low volume at vector %d/%d at %0.3f s/vector', 
                     i, 
                     len(vectors), 
                     toc-tic)
    to_origin = np.dot(rotation_Z, to_2D)
    transformed = transform_points(hull.vertices, to_origin)
    box_center = (transformed.min(axis=0) + transformed.ptp(axis=0)*.5)
    to_origin[0:3, 3] = -box_center
    return to_origin, extents
