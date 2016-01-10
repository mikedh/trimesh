'''
Create meshes from primitives, or with operations. 
'''

from .base      import Trimesh
from .constants import log
from .geometry  import faces_to_edges
from .grouping  import group_rows, unique_rows
from .util      import three_dimensionalize, append_faces

import numpy as np
from collections import deque
from shapely.geometry import Polygon

def extrude_polygon(polygon, 
                    height,
                    fix_normals=True,
                    **kwargs):
    '''
    Turn a shapely.geometry Polygon object and a height (float)
    into a watertight Trimesh object. 
    '''
    # create a 2D triangulation of the shapely polygon
    vertices, faces = triangulate_polygon(polygon, **kwargs)
    
    # stack the (n,3) faces into (3*n, 2) edges
    edges = np.sort(faces_to_edges(faces), axis=1)
    # edges which only occur once are on the boundary of the polygon
    # since the triangulation may have subdivided the boundary of the
    # shapely polygon, we need to find it again
    edges_unique = group_rows(edges, require_count = 1)

    # (n, 2, 2) set of line segments (positions, not references)
    boundary = vertices[edges[edges_unique]]

    # we are creating two vertical  triangles for every 2D line segment
    # on the boundary of the 2D triangulation
    vertical = np.tile(boundary.reshape((-1,2)), 2).reshape((-1,2))
    vertical = np.column_stack((vertical, 
                                np.tile([0,height,0,height], len(boundary))))
    vertical_faces  = np.tile([0,1,2,2,1,3], (len(boundary), 1))
    vertical_faces += np.arange(len(boundary)).reshape((-1,1)) * 4
    vertical_faces  = vertical_faces.reshape((-1,3))

    # stack the (n,2) vertices with zeros to make them (n, 3)
    vertices_3D  = three_dimensionalize(vertices, return_2D = False)
    
    # a sequence of zero- indexed faces, which will then be appended
    # with offsets to create the final mesh
    faces_seq    = [faces, faces.copy()[:,::-1], vertical_faces]
    vertices_seq = [vertices_3D, (vertices_3D.copy() + [0.0, 0, height]), vertical]

    mesh = Trimesh(*append_faces(vertices_seq, faces_seq), process=True)
    # the winding and normals of our mesh are arbitrary, although now we are
    # watertight so we can traverse the mesh and fix winding and normals 
    if fix_normals: mesh.fix_normals()
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

