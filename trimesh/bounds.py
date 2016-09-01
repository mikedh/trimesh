import numpy as np
import time

from .constants  import log
from .geometry   import rotation_2D_to_3D
from .points     import project_to_plane, transform_points

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
    
    points = np.asanyarray(points)
    points_unique = points[grouping.unique_rows(points)[0]]
    c = spatial.ConvexHull(points_unique, qhull_options='QbB')
    # (n,2,3) line segments
    hull = c.points[c.simplices] 
    hull_points = c.points[c.vertices]
    
    edge_vectors = util.unitize(np.diff(hull, axis=1).reshape((-1,2)))
    perp_vectors = np.fliplr(edge_vectors) * [-1.0,1.0]
    
    x = np.dot(edge_vectors, hull_points.T)
    y = np.dot(perp_vectors, hull_points.T)
    bounds = np.column_stack((x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)))
    
    extents  = np.diff(bounds.reshape((-1,2,2)), axis=1).reshape((-1,2))
    area     = np.product(extents, axis=1)
    area_min = area.argmin()

    #(2,) float of OBB rectangle size
    rectangle = extents[area_min]

    # find the (3,3) homogenous transformation which moves the input points
    # to have a bounding box centered at the origin
    offset = -bounds[area_min][0:2] - (rectangle * .5)
    theta  = np.arctan2(*edge_vectors[area_min][::-1])
    transform = util.transformation_2D(offset, theta)

    return transform, rectangle

def oriented_bounds(obj, angle_tol=1e-6):
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
    if hasattr(obj, 'convex_hull_raw'):
        # if we have been passed a mesh, use its existing convex hull to pull from 
        # cache rather than recomputing. This version of the cached convex hull has 
        # normals pointing in arbitrary directions (straight from qhull)
        # using this avoids having to compute the expensive corrected normals
        # that mesh.convex_hull uses since normal directions don't matter here
        vertices     = obj.convex_hull_raw.vertices
        face_normals = obj.convex_hull_raw.face_normals
    elif util.is_sequence(obj):
        points = np.asanyarray(obj)
        if util.is_shape(points, (-1,2)):
            return oriented_bounds_2D(points)
        elif util.is_shape(points, (-1,3)):
            hull_obj = spatial.ConvexHull(points)
            vertices = hull_obj.points[hull_obj.vertices]
            face_normals, valid  = triangles.normals(hull_obj.points[hull_obj.simplices])
        else:
            raise ValueError('Points are not (n,3) or (n,2)!')
    else:
        raise ValueError('Oriented bounds must be passed a mesh or a set of points!')
        
    vectors = grouping.group_vectors(face_normals, 
                                     angle=angle_tol,
                                     include_negative=True)[0]
    min_volume = np.inf
    tic = time.time()
    for i, normal in enumerate(vectors):
        projected, to_3D = project_to_plane(vertices,
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
    transformed = transform_points(vertices, to_origin)
    box_center = (transformed.min(axis=0) + transformed.ptp(axis=0)*.5)
    to_origin[0:3, 3] = -box_center
 
    log.debug('oriented_bounds checked %d vectors in %0.4fs',
              len(vectors),
              time.time()-tic)
    return to_origin, extents

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
        projected = transform_points(hull, to_2D)
        height = projected[:,2].ptp()
        # in degenerate cases return as infinite volume
        try:    center_2D, radius = minimum_nsphere(projected[:,0:2])
        except: return np.inf

        volume = np.pi * height * (radius ** 2)
        if return_data:
            center_3D = np.append(center_2D, projected[:,2].min() + (height * .5))
            transform = np.dot(np.linalg.inv(to_2D), 
                               transformations.translation_matrix(center_3D))
            return transform, radius, height
        return volume
        
    hull = convex.hull_points(obj)
    if not util.is_shape(hull, (-1,3)):
        raise ValueError('Input must be reducable to 3D points!')
    
    # sample a hemisphere so local hill climbing can do its thing
    samples = util.grid_linspace_2D([[0,0], [np.pi, np.pi]], sample_count)
    tic = [time.time()]
    # the best vector in (2,) spherical coordinates
    best = samples[np.argmin([volume_from_angles(i) for i in samples])]
    tic.append(time.time())
    
    # since we already explored the global space, set the bounds to be
    # just around the sample that had the lowest volume
    step = 2*np.pi/sample_count
    bounds = [(best[0]-step, best[0]+step),
              (best[1]-step, best[1]+step)]
    
    # run the optimization
    r = optimize.minimize(volume_from_angles, best, tol=angle_tol, method='SLSQP', bounds=bounds)
    tic.append(time.time())
    
    log.info('Performed search in %f and minimize in %f', *np.diff(tic))
    
    # actually chunk the information about the cylinder
    transform, radius, height = volume_from_angles(r['x'], return_data=True)
    result = {'transform' : transform, 
              'radius'    : radius, 
              'height'    : height}
    return result  
