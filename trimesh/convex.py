'''
trimesh.py

Library for importing and doing simple operations on triangular meshes.
'''

import numpy as np

from .constants import tol, log

from . import util
from . import triangles


try:
    from scipy import spatial
except ImportError:
    log.warning('Scipy import failed!')


def convex_hull(obj, qhull_options='QbB Pp'):
    '''
    Get a new Trimesh object representing the convex hull of the
    current mesh, with proper normals and watertight. 
    Requires scipy >.12.

    Argments
    --------
    obj: Trimesh object OR 
         (n,3) float, cartesian points

    Returns
    --------
    convex: Trimesh object of convex hull
    '''
    from .base import Trimesh

    if isinstance(obj, Trimesh):
        points = obj.vertices.view(np.ndarray)
    else:
        points = np.asanyarray(obj, dtype=np.float64)
        if not util.is_shape(points, (-1, 3)):
            raise ValueError('Object must be Trimesh or (n,3) points!')

    c = spatial.ConvexHull(points.reshape((-1, 3)),
                           qhull_options=qhull_options)

    # hull object doesn't remove unreferenced vertices
    # create a mask to re- index faces for only referenced vertices
    vid = np.sort(c.vertices)
    mask = np.zeros(len(c.points), dtype=np.int64)
    mask[vid] = np.arange(len(vid))
    # remove unreferenced vertices here
    faces = mask[c.simplices].copy()
    vertices = c.points[vid].copy()

    # qhull returns faces with random winding
    # calculate the returned normal of each face
    crosses = triangles.cross(vertices[faces])
    normals, valid = triangles.normals(crosses=crosses)

    # remove degenerate faces
    faces = faces[valid]
    crosses = crosses[valid]

    # calcalate each triangles area and cartesian center point
    triangles_area = triangles.area(crosses=crosses, sum=False)
    triangles_center = vertices[faces].mean(axis=1)

    # since the convex hull is (very hopefully) convex, the vector from
    # the centroid to the center of each face
    # should have a positive dot product with the normal of that face
    # if it doesn't it is probably backwards
    # note that this sometimes gets screwed up by precision issues
    centroid = np.average(triangles_center,
                          weights=triangles_area,
                          axis=0)
    test_vector = vertices[faces[:, 0]] - centroid
    backwards = util.diagonal_dot(normals,
                                  test_vector) < 0.0

    # flip the winding and normals to be outward facing
    faces[backwards] = np.fliplr(faces[backwards])
    normals[backwards] *= -1.0

    # save the work we did to the cache so it doesn't have to be recomputed
    initial_cache = {'triangles_cross': crosses,
                     'triangles_center': triangles_center,
                     'area_faces': triangles_area,
                     'centroid': centroid}

    # create the Trimesh object for the convex hull
    convex = Trimesh(vertices=vertices,
                     faces=faces,
                     face_normals=normals,
                     initial_cache=initial_cache,
                     process=True)

    # we did the gross case above, but sometimes precision issues
    # leave some faces backwards anyway
    # this call will exit early if the winding is consistent
    # and if not will (slowly) fix it by traversing the adjacency graph
    convex.fix_normals()

    # sometimes the QbB option will cause unrecoverable precision issues
    # so try the hull again without it
    # check for qhull_options is None to avoid infinite recursion
    if (qhull_options is not None and
            not convex.is_winding_consistent):
        return convex_hull(convex, qhull_options=None)

    return convex


def is_convex(mesh):
    '''
    Test if a mesh is convex by projecting the vertices of
    a triangle onto the normal of its adjacent face.

    Parameters
    ----------
    mesh: Trimesh object

    Returns
    ----------
    convex: bool, is the mesh convex or not
    '''
    # normals and origins from the first column of face adjacency
    normals = mesh.face_normals[mesh.face_adjacency[:, 0]]
    # one of the vertices on the shared edge
    origins = mesh.vertices[mesh.face_adjacency_edges[:, 0]]

    # faces from the second column of face adjacency
    faces = mesh.faces[mesh.face_adjacency[:, 1]]
    shared = np.logical_or(faces == mesh.face_adjacency_edges[:, 0].reshape((-1, 1)),
                           faces == mesh.face_adjacency_edges[:, 1].reshape((-1, 1)))
    vertex_other = mesh.vertices[faces[np.logical_not(shared)]]

    vector_other = vertex_other - origins
    length_other = (vector_other ** 2).sum(axis=1) ** .5

    dots = util.diagonal_dot(vector_other,
                             normals)
    convex = (dots < tol.merge).all()
    return bool(convex)


def planar_hull(points, normal, origin=None, input_convex=False):
    '''
    Find the convex outline of a set of points projected to a plane.

    Parameters
    -----------
    points: (n,3) float, input points
    normal: (3) float vector, normal vector of plane
    origin: (3) float, location of plane origin
    input_convex: bool, if True we assume the input points are already from
                  a convex hull which provides a speedup.

    Returns
    -----------
    hull_lines: (n,2,2) set of unordered line segments
    T:          (4,4) float, transformation matrix
    '''
    from .points import project_to_plane

    if origin is None:
        origin = np.zeros(3)
    if not input_convex:
        pass
    planar, T = project_to_plane(points,
                                 plane_normal=normal,
                                 plane_origin=origin,
                                 return_planar=False,
                                 return_transform=True)
    hull_edges = spatial.ConvexHull(planar[:, 0:2]).simplices
    hull_lines = planar[hull_edges]
    planar_z = planar[:, 2]
    height = np.array([planar_z.min(),
                       planar_z.max()])
    return hull_lines, T, height


def hull_points(obj):
    '''
    Try to extract a convex set of points from multiple input formats.

    Parameters
    ---------
    obj: Trimesh object
         (n,d) points
         (m,) Trimesh objects

    Returns
    --------
    points: (o,d) convex set of points
    '''
    if hasattr(obj, 'convex_hull'):
        points = obj.convex_hull.vertices
    elif util.is_sequence(obj):
        initial = np.asanyarray(obj)
        if len(initial.shape) != 2:
            raise ValueError('Points must be (n, dimension)!')
        hull = spatial.ConvexHull(initial, qhull_options='QbB Pp')
        points = hull.points[hull.vertices]
    else:
        raise ValueError('Can\'t extract hull points from %s',
                         obj.__class__.__name__)
    return points
