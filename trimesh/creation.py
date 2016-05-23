'''
Create meshes from primitives, or with operations. 
'''

from .base      import Trimesh
from .constants import log, tol
from .triangles import normals
from .geometry  import faces_to_edges
from .grouping  import group_rows, unique_rows

from . import util

import numpy as np

from collections import deque

try:    
    from shapely.geometry import Polygon
except ImportError: 
    pass

def validate_polygon(obj):
    if util.is_instance_named(obj, 'Polygon'):
        polygon = obj
    elif util.is_shape(obj, (-1,2)):
        polygon = Polygon(obj)
    else:
        raise ValueError('Input not a polygon!')
        
    if (not polygon.is_valid or
        polygon.area < tol.zero):
        raise ValueError('Polygon is zero- area or invalid!')
    return polygon
    

def extrude_polygon(polygon,
                    height,
                    **kwargs):    
    # create a 2D triangulation of the shapely polygon
    vertices, faces = triangulate_polygon(polygon, **kwargs)
    mesh = extrude_triangulation(vertices = vertices, 
                                 faces    = faces,
                                 height   = height, 
                                 **kwargs)
    return mesh

def extrude_triangulation(vertices, 
                          faces,
                          height,
                          **kwargs):
    '''
    Turn a shapely.geometry Polygon object and a height (float)
    into a watertight Trimesh object. 
    '''
    # make sure triangulation winding is pointing up
    normal_test = normals([util.three_dimensionalize(vertices[faces[0]])[1]])[0]
    if np.dot(normal_test, [0,0,1]) < 0:
        faces = np.fliplr(faces)
    
    # stack the (n,3) faces into (3*n, 2) edges
    edges = faces_to_edges(faces)
    edges_sorted = np.sort(edges, axis=1)
    # edges which only occur once are on the boundary of the polygon
    # since the triangulation may have subdivided the boundary of the
    # shapely polygon, we need to find it again
    edges_unique = group_rows(edges_sorted, require_count = 1)

    # (n, 2, 2) set of line segments (positions, not references)
    boundary = vertices[edges[edges_unique]]

    # we are creating two vertical  triangles for every 2D line segment
    # on the boundary of the 2D triangulation
    vertical = np.tile(boundary.reshape((-1,2)), 2).reshape((-1,2))
    vertical = np.column_stack((vertical, 
                                np.tile([0,height,0,height], len(boundary))))
    vertical_faces  = np.tile([3,1,2,2,1,0], (len(boundary), 1))
    vertical_faces += np.arange(len(boundary)).reshape((-1,1)) * 4
    vertical_faces  = vertical_faces.reshape((-1,3))

    # stack the (n,2) vertices with zeros to make them (n, 3)
    vertices_3D  = util.three_dimensionalize(vertices, return_2D = False)
    
    # a sequence of zero- indexed faces, which will then be appended
    # with offsets to create the final mesh
    faces_seq    = [faces[:,::-1], faces.copy(), vertical_faces]
    vertices_seq = [vertices_3D, (vertices_3D.copy() + [0.0, 0, height]), vertical]

    mesh = Trimesh(*util.append_faces(vertices_seq, faces_seq), process=True)

    return mesh

def triangulate_polygon(polygon, **kwargs):
    '''
    Given a shapely polygon, create a triangulation using meshpy.triangle

    Arguments
    ---------
    polygon: Shapely.geometry.Polygon
    kwargs: passed directly to meshpy.triangle.build:
            triangle.build(mesh_info, 
                           verbose=False, 
                           refinement_func=None, 
                           attributes=False, 
                           volume_constraints=True, 
                           max_volume=None, 
                           allow_boundary_steiner=True, 
                           allow_volume_steiner=True, 
                           quality_meshing=True, 
                           generate_edges=None, 
                           generate_faces=False, 
                           min_angle=None)
    Returns
    --------
    mesh_vertices: (n, 2) float array of 2D points
    mesh_faces:    (n, 3) int array of vertex indicies representing triangles
    '''

    # do the import here, as sometimes this import can segfault python
    # which is not catchable with a try/except block
    import meshpy.triangle as triangle

    def round_trip(start, length):
        '''
        Given a start index and length, create a series of (n, 2) edges which
        create a closed traversal. 

        Example:
        start, length = 0, 3
        returns:  [(0,1), (1,2), (2,0)]
        '''
        tiled = np.tile(np.arange(start, start+length).reshape((-1,1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1,2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled

    def add_boundary(boundary, start):
        # coords is an (n, 2) ordered list of points on the polygon boundary
        # the first and last points are the same, and there are no
        # guarentees on points not being duplicated (which will 
        # later cause meshpy/triangle to shit a brick)
        coords  = np.array(boundary.coords)
        # find indices points which occur only once, and sort them
        # to maintain order
        unique  = np.sort(unique_rows(coords)[0])
        cleaned = coords[unique]

        vertices.append(cleaned)
        facets.append(round_trip(start, len(cleaned)))

        # holes require points inside the region of the hole, which we find
        # by creating a polygon from the cleaned boundary region, and then
        # using a representative point. You could do things like take the mean of 
        # the points, but this is more robust (to things like concavity), if slower. 
        test = Polygon(cleaned)
        holes.append(np.array(test.representative_point().coords)[0])

        return len(cleaned)

    #sequence of (n,2) points in space
    vertices = deque()
    #sequence of (n,2) indices of vertices
    facets   = deque()
    #list of (2) vertices in interior of hole regions
    holes    = deque()

    start = add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        try: start += add_boundary(interior, start)
        except: 
            log.warn('invalid interior, continuing')
            continue

    # create clean (n,2) float array of vertices
    # and (m, 2) int array of facets
    # by stacking the sequence of (p,2) arrays
    vertices = np.vstack(vertices)
    facets   = np.vstack(facets)
    
    # holes in meshpy lingo are a (h, 2) list of (x,y) points
    # which are inside the region of the hole
    # we added a hole for the exterior, which we slice away here
    holes    = np.array(holes)[1:]

    # call meshpy.triangle on our cleaned representation of the Shapely polygon
    info = triangle.MeshInfo()
    info.set_points(vertices)
    info.set_facets(facets)
    info.set_holes(holes)

    # uses kwargs
    mesh = triangle.build(info, **kwargs)
  
    mesh_vertices = np.array(mesh.points)
    mesh_faces    = np.array(mesh.elements)

    return mesh_vertices, mesh_faces

def box():
    '''
    Return a unit cube, centered at the origin with edges of length 1.0
    '''
    vertices = [0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1] 
    vertices = np.array(vertices, dtype=np.float64).reshape((-1,3))
    vertices -= 0.5

    faces = [1,3,0,4,1,0,0,3,2,2,4,0,1,7,3,5,1,4,
             5,7,1,3,7,2,6,4,2,2,7,6,6,5,4,7,5,6] 
    faces = np.array(faces, dtype=np.int64).reshape((-1,3))

    face_normals = [-1,0,0,0,-1,0,-1,0,0,0,0,-1,0,0,1,0,-1,
                    0,0,0,1,0,1,0,0,0,-1,0,1,0,1,0,0,1,0,0]
    face_normals = np.array(face_normals, dtype=np.float64).reshape(-1,3)

    box = Trimesh(vertices = vertices, 
                  faces = faces,
                  face_normals = face_normals)
    return box

def icosahedron():
    '''
    Create an icosahedron, a 20 faced polyhedron.
    '''
    t = (1.0 + 5.0**.5) / 2.0
    v = [-1,t,0,1,t,0,-1,-t,0,1,-t,0,0,-1,t,0,1,t,
         0,-1,-t,0,1,-t,t,0,-1,t,0,1,-t,0,-1,-t,0,1]
    f = [0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,
         1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,
         3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,
         4,9,5,2,4,11,6,2,10,8,6,7,9,8,1]
    v = np.reshape(v, (-1,3))
    f = np.reshape(f, (-1,3))
    m = Trimesh(v,f)
    return m

def icosphere(subdivisions=3):
    '''
    Create an isophere of radius 1.0 centered at the origin.
    '''
    def refine_spherical():
        vectors = ico.vertices
        scalar  = (vectors ** 2).sum(axis=1)**.5
        unit    = vectors / scalar.reshape((-1,1))
        offset  = 1.0 - scalar
        ico.vertices += unit * offset.reshape((-1,1))
    ico = icosahedron()
    ico._validate = False
    for j in range(subdivisions): 
        ico.subdivide()
        refine_spherical()
    ico._validate = True
    return ico

def cylinder(radius, height, sections=32):
    '''
    Create a mesh of a cylinder along Z centered at the origin.

    Arguments
    ----------
    radius: float, the radius of the cylinder
    height: float, the height of the cylinder
    sections: int, how many pie wedges should the cylinder be meshed as

    Returns
    ----------
    cylinder: Trimesh, resulting mesh
    '''
    theta = np.linspace(0, np.pi*2, sections)

    vertices = np.column_stack((np.sin(theta), np.cos(theta))) * radius
    vertices[0] = [0,0]

    index = np.arange(1, len(vertices)+1).reshape((-1,1))
    index[-1] = 1

    faces = np.tile(index,(1,2)).reshape(-1)[1:-1].reshape((-1,2))
    faces = np.column_stack((np.zeros(len(faces), dtype=np.int), faces))

    cylinder = extrude_triangulation(vertices = vertices, 
                                     faces  = faces, 
                                     height = height)
    cylinder.vertices[:,2] -= height * .5
    
    return cylinder
