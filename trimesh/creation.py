"""
creation.py
--------------

Create meshes from primitives, or with operations.
"""

from .base import Trimesh
from .constants import log, tol
from .triangles import normals
from .geometry import faces_to_edges, align_vectors

from . import util
from . import grouping
from . import transformations

import numpy as np
import collections

try:
    # shapely is a soft dependency
    from shapely.geometry import Polygon
    from shapely.wkb import loads as load_wkb
except BaseException:
    # shapely will sometimes raise OSErrors
    # on import rather than just ImportError
    log.warning('shapely.geometry.Polygon not available!',
                exc_info=True)


def validate_polygon(obj):
    """
    Make sure an input can be returned as a valid polygon.

    Parameters
    -------------
    obj : shapely.geometry.Polygon, str (wkb), or (n, 2) float
      Object which might be a polygon

    Returns
    ------------
    polygon : shapely.geometry.Polygon
      Valid polygon object

    Raises
    -------------
    ValueError
      If a valid finite- area polygon isn't available
    """
    if isinstance(obj, Polygon):
        polygon = obj
    elif util.is_shape(obj, (-1, 2)):
        polygon = Polygon(obj)
    elif util.is_string(obj):
        polygon = load_wkb(obj)
    else:
        raise ValueError('Input not a polygon!')

    if (not polygon.is_valid or
            polygon.area < tol.zero):
        raise ValueError('Polygon is zero- area or invalid!')
    return polygon


def extrude_polygon(polygon,
                    height,
                    **kwargs):
    """
    Extrude a 2D shapely polygon into a 3D mesh

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
      2D geometry to extrude
    height : float
      Distance to extrude polygon along Z
    **kwargs:
        passed to Trimesh

    Returns
    ----------
    mesh : trimesh.Trimesh
      Resulting extrusion as watertight body
    """
    vertices, faces = triangulate_polygon(polygon, **kwargs)
    mesh = extrude_triangulation(vertices=vertices,
                                 faces=faces,
                                 height=height,
                                 **kwargs)
    return mesh


def sweep_polygon(polygon,
                  path,
                  angles=None,
                  **kwargs):
    """
    Extrude a 2D shapely polygon into a 3D mesh along an
    arbitrary 3D path. Doesn't handle sharp curvature.


    Parameters
    ----------
    polygon : shapely.geometry.Polygon
      Profile to sweep along path
    path : (n, 3) float
      A path in 3D
    angles :  (n,) float
      Optional rotation angle relative to prior vertex
      at each vertex

    Returns
    -------
    mesh : trimesh.Trimesh
      Geometry of result
    """

    path = np.asanyarray(path, dtype=np.float64)
    if not util.is_shape(path, (-1, 3)):
        raise ValueError('Path must be (n, 3)!')

    # Extract 2D vertices and triangulation
    verts_2d = np.array(polygon.exterior)[:-1]
    base_verts_2d, faces_2d = triangulate_polygon(polygon, **kwargs)
    n = len(verts_2d)

    # Create basis for first planar polygon cap
    x, y, z = util.generate_basis(path[0] - path[1])
    tf_mat = np.ones((4, 4))
    tf_mat[:3, :3] = np.c_[x, y, z]
    tf_mat[:3, 3] = path[0]

    # Compute 3D locations of those vertices
    verts_3d = np.c_[verts_2d, np.zeros(n)]
    verts_3d = transformations.transform_points(verts_3d, tf_mat)
    base_verts_3d = np.c_[base_verts_2d,
                          np.zeros(len(base_verts_2d))]
    base_verts_3d = transformations.transform_points(base_verts_3d,
                                                     tf_mat)

    # keep matching sequence of vertices and 0- indexed faces
    vertices = [base_verts_3d]
    faces = [faces_2d]

    # Compute plane normals for each turn --
    # each turn induces a plane halfway between the two vectors
    v1s = util.unitize(path[1:-1] - path[:-2])
    v2s = util.unitize(path[1:-1] - path[2:])
    norms = np.cross(np.cross(v1s, v2s), v1s + v2s)
    norms[(norms == 0.0).all(1)] = v1s[(norms == 0.0).all(1)]
    norms = util.unitize(norms)
    final_v1 = util.unitize(path[-1] - path[-2])
    norms = np.vstack((norms, final_v1))
    v1s = np.vstack((v1s, final_v1))

    # Create all side walls by projecting the 3d vertices into each plane
    # in succession
    for i in range(len(norms)):
        verts_3d_prev = verts_3d

        # Rotate if needed
        if angles is not None:
            tf_mat = transformations.rotation_matrix(angles[i],
                                                     norms[i],
                                                     path[i])
            verts_3d_prev = transformations.transform_points(verts_3d_prev,
                                                             tf_mat)

        # Project vertices onto plane in 3D
        ds = np.einsum('ij,j->i', (path[i + 1] - verts_3d_prev), norms[i])
        ds = ds / np.dot(v1s[i], norms[i])

        verts_3d_new = np.einsum('i,j->ij', ds, v1s[i]) + verts_3d_prev

        # Add to face and vertex lists
        new_faces = [[i + n, (i + 1) % n, i] for i in range(n)]
        new_faces.extend([[(i - 1) % n + n, i + n, i] for i in range(n)])

        # save faces and vertices into a sequence
        faces.append(np.array(new_faces))
        vertices.append(np.vstack((verts_3d, verts_3d_new)))

        verts_3d = verts_3d_new

    # do the main stack operation from a sequence to (n,3) arrays
    # doing one vstack provides a substantial speedup by
    # avoiding a bunch of temporary  allocations
    vertices, faces = util.append_faces(vertices, faces)

    # Create final cap
    x, y, z = util.generate_basis(path[-1] - path[-2])
    vecs = verts_3d - path[-1]
    coords = np.c_[np.einsum('ij,j->i', vecs, x),
                   np.einsum('ij,j->i', vecs, y)]
    base_verts_2d, faces_2d = triangulate_polygon(Polygon(coords))
    base_verts_3d = (np.einsum('i,j->ij', base_verts_2d[:, 0], x) +
                     np.einsum('i,j->ij', base_verts_2d[:, 1], y)) + path[-1]
    faces = np.vstack((faces, faces_2d + len(vertices)))
    vertices = np.vstack((vertices, base_verts_3d))

    return Trimesh(vertices, faces)


def extrude_triangulation(vertices,
                          faces,
                          height,
                          **kwargs):
    """
    Turn a 2D triangulation into a watertight Trimesh.

    Parameters
    ----------
    vertices : (n, 2) float
      2D vertices
    faces : (m, 3) int
      Triangle indexes of vertices
    height : float
      Distance to extrude triangulation
    **kwargs:
        passed to Trimesh

    Returns
    ---------
    mesh : trimesh.Trimesh
      Mesh created from extrusion
    """
    vertices = np.asanyarray(vertices, dtype=np.float64)
    height = float(height)
    faces = np.asanyarray(faces, dtype=np.int64)

    if not util.is_shape(vertices, (-1, 2)):
        raise ValueError('Vertices must be (n,2)')
    if not util.is_shape(faces, (-1, 3)):
        raise ValueError('Faces must be (n,3)')
    if np.abs(height) < tol.merge:
        raise ValueError('Height must be nonzero!')

    # make sure triangulation winding is pointing up
    normal_test = normals(
        [util.stack_3D(vertices[faces[0]])])[0]

    normal_dot = np.dot(normal_test,
                        [0.0, 0.0, np.sign(height)])[0]

    # make sure the triangulation is aligned with the sign of
    # the height we've been passed
    if normal_dot < 0.0:
        faces = np.fliplr(faces)

    # stack the (n,3) faces into (3*n, 2) edges
    edges = faces_to_edges(faces)
    edges_sorted = np.sort(edges, axis=1)
    # edges which only occur once are on the boundary of the polygon
    # since the triangulation may have subdivided the boundary of the
    # shapely polygon, we need to find it again
    edges_unique = grouping.group_rows(edges_sorted, require_count=1)

    # (n, 2, 2) set of line segments (positions, not references)
    boundary = vertices[edges[edges_unique]]

    # we are creating two vertical  triangles for every 2D line segment
    # on the boundary of the 2D triangulation
    vertical = np.tile(boundary.reshape((-1, 2)), 2).reshape((-1, 2))
    vertical = np.column_stack((vertical,
                                np.tile([0, height, 0, height],
                                        len(boundary))))
    vertical_faces = np.tile([3, 1, 2, 2, 1, 0],
                             (len(boundary), 1))
    vertical_faces += np.arange(len(boundary)).reshape((-1, 1)) * 4
    vertical_faces = vertical_faces.reshape((-1, 3))

    # stack the (n,2) vertices with zeros to make them (n, 3)
    vertices_3D = util.stack_3D(vertices)

    # a sequence of zero- indexed faces, which will then be appended
    # with offsets to create the final mesh
    faces_seq = [faces[:, ::-1],
                 faces.copy(),
                 vertical_faces]
    vertices_seq = [vertices_3D,
                    vertices_3D.copy() + [0.0, 0, height],
                    vertical]

    mesh = Trimesh(*util.append_faces(vertices_seq,
                                      faces_seq),
                   process=True,
                   **kwargs)

    assert mesh.volume > 0.0

    return mesh


def triangulate_polygon(polygon,
                        triangle_args='pq30',
                        engine='auto',
                        **kwargs):
    """
    Given a shapely polygon create a triangulation using one
    of the python interfaces to triangle.c:
    > pip install meshpy
    > pip install triangle

    Parameters
    ---------
    polygon : Shapely.geometry.Polygon
        Polygon object to be triangulated
    triangle_args : str
        Passed to triangle.triangulate
    engine : str
        'meshpy', 'triangle', or 'auto'
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
    --------------
    vertices : (n, 2) float
       Points in space
    faces :    (n, 3) int
       Index of vertices that make up triangles
    """

    # turn the polygon in to vertices, segments, and hole points
    arg = _polygon_to_kwargs(polygon)

    try:
        if str(engine).strip() in ['auto', 'triangle']:
            from triangle import triangulate
            result = triangulate(arg, triangle_args)
            return result['vertices'], result['triangles']
    except ImportError:
        # no `triangle` so move on to `meshpy`
        pass
    except BaseException as E:
        # if we see an exception log it and move on
        log.error('failed to triangulate using triangle!',
                  exc_info=True)
        # if we are running unit tests exit here and fail
        if tol.strict:
            raise E

    # do the import here, as sometimes this import can segfault
    # which is not catchable with a try/except block
    from meshpy import triangle
    # call meshpy.triangle on our cleaned representation
    info = triangle.MeshInfo()
    info.set_points(arg['vertices'])
    info.set_facets(arg['segments'])
    # not all polygons have holes
    if 'holes' in arg:
        info.set_holes(arg['holes'])
    # build mesh and pass kwargs to triangle
    mesh = triangle.build(info, **kwargs)
    # (n, 2) float vertices
    vertices = np.array(mesh.points, dtype=np.float64)
    # (m, 3) int faces
    faces = np.array(mesh.elements, dtype=np.int64)

    return vertices, faces


def _polygon_to_kwargs(polygon):
    """
    Given a shapely polygon generate the data to pass to
    the triangle mesh generator

    Parameters
    ---------
    polygon : Shapely.geometry.Polygon
      Input geometry

    Returns
    --------
    result : dict
      Has keys: vertices, segments, holes
    """

    if not polygon.is_valid:
        raise ValueError('invalid shapely polygon passed!')

    def round_trip(start, length):
        """
        Given a start index and length, create a series of (n, 2) edges which
        create a closed traversal.

        Examples
        ---------
        start, length = 0, 3
        returns:  [(0,1), (1,2), (2,0)]
        """
        tiled = np.tile(np.arange(start, start + length).reshape((-1, 1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1, 2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled

    def add_boundary(boundary, start):
        # coords is an (n, 2) ordered list of points on the polygon boundary
        # the first and last points are the same, and there are no
        # guarantees on points not being duplicated (which will
        # later cause meshpy/triangle to shit a brick)
        coords = np.array(boundary.coords)
        # find indices points which occur only once, and sort them
        # to maintain order
        unique = np.sort(grouping.unique_rows(coords)[0])
        cleaned = coords[unique]

        vertices.append(cleaned)
        facets.append(round_trip(start, len(cleaned)))

        # holes require points inside the region of the hole, which we find
        # by creating a polygon from the cleaned boundary region, and then
        # using a representative point. You could do things like take the mean of
        # the points, but this is more robust (to things like concavity), if
        # slower.
        test = Polygon(cleaned)
        holes.append(np.array(test.representative_point().coords)[0])

        return len(cleaned)

    # sequence of (n,2) points in space
    vertices = collections.deque()
    # sequence of (n,2) indices of vertices
    facets = collections.deque()
    # list of (2) vertices in interior of hole regions
    holes = collections.deque()

    start = add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        try:
            start += add_boundary(interior, start)
        except BaseException:
            log.warn('invalid interior, continuing')
            continue

    # create clean (n,2) float array of vertices
    # and (m, 2) int array of facets
    # by stacking the sequence of (p,2) arrays
    vertices = np.vstack(vertices)
    facets = np.vstack(facets).tolist()

    # shapely polygons can include a Z component
    # strip it out for the triangulation
    if vertices.shape[1] == 3:
        vertices = vertices[:, :2]

    result = {'vertices': vertices,
              'segments': facets}
    # holes in meshpy lingo are a (h, 2) list of (x,y) points
    # which are inside the region of the hole
    # we added a hole for the exterior, which we slice away here
    holes = np.array(holes)[1:]
    if len(holes) > 0:
        result['holes'] = holes

    return result


def box(extents=None, transform=None, **kwargs):
    """
    Return a cuboid.

    Parameters
    ------------
    extents : float, or (3,) float
      Edge lengths
    transform: (4, 4) float
      Transformation matrix
    **kwargs:
        passed to Trimesh to create box

    Returns
    ------------
    box : trimesh.Trimesh
      Cuboid geometry in space
    """
    vertices = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    vertices = np.array(vertices,
                        order='C',
                        dtype=np.float64).reshape((-1, 3))
    vertices -= 0.5

    if extents is not None:
        extents = np.asanyarray(extents, dtype=np.float64)
        if extents.shape != (3,):
            raise ValueError('Extents must be (3,)!')
        vertices *= extents

    faces = [1, 3, 0, 4, 1, 0, 0, 3, 2, 2, 4, 0, 1, 7, 3, 5, 1, 4,
             5, 7, 1, 3, 7, 2, 6, 4, 2, 2, 7, 6, 6, 5, 4, 7, 5, 6]
    faces = np.array(faces,
                     order='C', dtype=np.int64).reshape((-1, 3))

    face_normals = [-1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 1, 0, -1,
                    0, 0, 0, 1, 0, 1, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    face_normals = np.asanyarray(face_normals,
                                 order='C',
                                 dtype=np.float64).reshape(-1, 3)

    box = Trimesh(vertices=vertices,
                  faces=faces,
                  face_normals=face_normals,
                  process=False,
                  **kwargs)
    if transform is not None:
        box.apply_transform(transform)

    return box


def icosahedron():
    """
    Create an icosahedron, a 20 faced polyhedron.

    Returns
    -------------
    ico : trimesh.Trimesh
      Icosahederon centered at the origin.
    """
    t = (1.0 + 5.0**.5) / 2.0
    vertices = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t,
                0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1]
    faces = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
             1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
             3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
             4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
    # scale vertices so each vertex radius is 1.0
    vertices = np.reshape(vertices, (-1, 3)) / np.sqrt(2.0 + t)
    faces = np.reshape(faces, (-1, 3))
    mesh = Trimesh(vertices=vertices,
                   faces=faces,
                   process=False)
    return mesh


def icosphere(subdivisions=3, radius=1.0):
    """
    Create an isophere centered at the origin.

    Parameters
    ----------
    subdivisions : int
      How many times to subdivide the mesh.
      Note that the number of faces will grow as function of
      4 ** subdivisions, so you probably want to keep this under ~5
    radius : float
      Desired radius of sphere

    Returns
    ---------
    ico : trimesh.Trimesh
      Meshed sphere
    """
    def refine_spherical():
        vectors = ico.vertices
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        ico.vertices += unit * offset.reshape((-1, 1))
    ico = icosahedron()
    ico._validate = False
    for j in range(subdivisions):
        ico = ico.subdivide()
        refine_spherical()
    ico._validate = True
    return ico


def uv_sphere(radius=1.0,
              count=[32, 32],
              theta=None,
              phi=None):
    """
    Create a UV sphere (latitude + longitude) centered at the
    origin. Roughly one order of magnitude faster than an
    icosphere but slightly uglier.

    Parameters
    ----------
    radius : float
      Radius of sphere
    count : (2,) int
      Number of latitude and longitude lines
    theta : (n,) float
      Optional theta angles in radians
    phi :   (n,) float
      Optional phi angles in radians

    Returns
    ----------
    mesh : trimesh.Trimesh
       Mesh of UV sphere with specified parameters
    """

    count = np.array(count, dtype=np.int)
    count += np.mod(count, 2)
    count[1] *= 2

    # generate vertices on a sphere using spherical coordinates
    if theta is None:
        theta = np.linspace(0, np.pi, count[0])
    if phi is None:
        phi = np.linspace(0, np.pi * 2, count[1])[:-1]
    spherical = np.dstack((np.tile(phi, (len(theta), 1)).T,
                           np.tile(theta, (len(phi), 1)))).reshape((-1, 2))
    vertices = util.spherical_to_vector(spherical) * radius

    # generate faces by creating a bunch of pie wedges
    c = len(theta)
    # a quad face as two triangles
    pairs = np.array([[c, 0, 1],
                      [c + 1, c, 1]])

    # increment both triangles in each quad face by the same offset
    incrementor = np.tile(np.arange(c - 1), (2, 1)).T.reshape((-1, 1))
    # create the faces for a single pie wedge of the sphere
    strip = np.tile(pairs, (c - 1, 1))
    strip += incrementor
    # the first and last faces will be degenerate since the first
    # and last vertex are identical in the two rows
    strip = strip[1:-1]

    # tile pie wedges into a sphere
    faces = np.vstack([strip + (i * c) for i in range(len(phi))])

    # poles are repeated in every strip, so a mask to merge them
    mask = np.arange(len(vertices))
    # the top pole are all the same vertex
    mask[0::c] = 0
    # the bottom pole are all the same vertex
    mask[c - 1::c] = c - 1

    # faces masked to remove the duplicated pole vertices
    # and mod to wrap to fill in the last pie wedge
    faces = mask[np.mod(faces, len(vertices))]

    # we save a lot of time by not processing again
    # since we did some bookkeeping mesh is watertight
    mesh = Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def capsule(height=1.0,
            radius=1.0,
            count=[32, 32]):
    """
    Create a mesh of a capsule, or a cylinder with hemispheric ends.

    Parameters
    ----------
    height : float
      Center to center distance of two spheres
    radius : float
      Radius of the cylinder and hemispheres
    count : (2,) int
      Number of sections on latitude and longitude

    Returns
    ----------
    capsule : trimesh.Trimesh
      Capsule geometry with:
        - cylinder axis is along Z
        - one hemisphere is centered at the origin
        - other hemisphere is centered along the Z axis at height
    """
    height = float(height)
    radius = float(radius)
    count = np.array(count, dtype=np.int)
    count += np.mod(count, 2)

    # create a theta where there is a double band around the equator
    # so that we can offset the top and bottom of a sphere to
    # get a nicely meshed capsule
    theta = np.linspace(0, np.pi, count[0])
    center = np.clip(np.arctan(tol.merge / radius),
                     tol.merge, np.inf)
    offset = np.array([-center, center]) + (np.pi / 2)
    theta = np.insert(theta,
                      int(len(theta) / 2),
                      offset)

    capsule = uv_sphere(radius=radius,
                        count=count,
                        theta=theta)

    top = capsule.vertices[:, 2] > tol.zero
    capsule.vertices[top] += [0, 0, height]

    return capsule


def cylinder(radius=1.0,
             height=1.0,
             sections=32,
             segment=None,
             transform=None,
             **kwargs):
    """
    Create a mesh of a cylinder along Z centered at the origin.

    Parameters
    ----------
    radius : float
      The radius of the cylinder
    height : float
      The height of the cylinder
    sections : int
      How many pie wedges should the cylinder have
    segment : (2, 3) float
      Endpoints of axis, overrides transform and height
    transform : (4, 4) float
      Transform to apply
    **kwargs:
        passed to Trimesh to create cylinder

    Returns
    ----------
    cylinder: trimesh.Trimesh
      Resulting mesh of a cylinder
    """

    if segment is not None:
        segment = np.asanyarray(segment, dtype=np.float64)
        if segment.shape != (2, 3):
            raise ValueError('segment must be 2 3D points!')
        vector = segment[1] - segment[0]
        # override height with segment length
        height = np.linalg.norm(vector)
        # point in middle of line
        midpoint = segment[0] + (vector * 0.5)
        # align Z with our desired direction
        rotation = align_vectors([0, 0, 1], vector)
        # translate to midpoint of segment
        translation = transformations.translation_matrix(midpoint)
        # compound the rotation and translation
        transform = np.dot(translation, rotation)

    # create a 2D pie out of wedges
    theta = np.linspace(0, np.pi * 2, sections)
    vertices = np.column_stack((np.sin(theta),
                                np.cos(theta))) * radius
    # the single vertex at the center of the circle
    # we're overwriting the duplicated start/end vertex
    vertices[0] = [0, 0]

    # whangle indexes into a triangulation of the pie wedges
    index = np.arange(1, len(vertices) + 1).reshape((-1, 1))
    index[-1] = 1
    faces = np.tile(index, (1, 2)).reshape(-1)[1:-1].reshape((-1, 2))
    faces = np.column_stack((np.zeros(len(faces), dtype=np.int), faces))

    # extrude the 2D triangulation into a Trimesh object
    cylinder = extrude_triangulation(vertices=vertices,
                                     faces=faces,
                                     height=height,
                                     **kwargs)
    # the extrusion was along +Z, so move the cylinder
    # center of mass back to the origin
    cylinder.vertices[:, 2] -= height * .5
    if transform is not None:
        # apply a transform here before any cache stuff is generated
        # and would have to be dumped after the transform is applied
        cylinder.apply_transform(transform)

    return cylinder


def annulus(r_min=1.0,
            r_max=2.0,
            height=1.0,
            sections=32,
            transform=None,
            **kwargs):
    """
    Create a mesh of an annular cylinder along Z,
    centered at the origin.

    Parameters
    ----------
    r_min : float
      The inner radius of the annular cylinder
    r_max : float
      The outer radius of the annular cylinder
    height : float
      The height of the annular cylinder
    sections : int
      How many pie wedges should the annular cylinder have
    **kwargs:
        passed to Trimesh to create annulus

    Returns
    ----------
    annulus : trimesh.Trimesh
      Mesh of annular cylinder
    """
    r_min = abs(float(r_min))
    r_max = abs(float(r_max))
    height = float(height)
    sections = int(sections)

    # if center radius is zero this is a cylinder
    if r_min < tol.merge:
        return cylinder(radius=r_max,
                        height=height,
                        sections=sections,
                        transform=transform)

    # create a 2D pie out of wedges
    theta = np.linspace(0, np.pi * 2, sections)[:-1]
    unit = np.column_stack((np.sin(theta),
                            np.cos(theta)))
    assert len(unit) == sections - 1

    vertices = np.vstack((unit * r_min,
                          unit * r_max))

    # one flattened triangulated quad covering one slice
    face = np.array([0, sections - 1, 1,
                     1, sections - 1, sections])

    # tile one quad into lots of quads
    faces = (np.tile(face, (sections - 1, 1)) +
             np.arange(sections - 1).reshape((-1, 1))).reshape((-1, 3))

    # stitch the last and first triangles with correct winding
    faces[-1] = [sections - 1, 0, sections - 2]

    # extrude the 2D profile into a mesh
    annulus = extrude_triangulation(vertices=vertices,
                                    faces=faces,
                                    height=height,
                                    **kwargs)

    # move the annulus so the centroid is at the origin
    annulus.vertices[:, 2] -= height * .5
    if transform is not None:
        # apply a transform here before any cache stuff is generated
        # and would have to be dumped after the transform is applied
        annulus.apply_transform(transform)

    return annulus


def random_soup(face_count=100):
    """
    Return random triangles as a Trimesh

    Parameters
    -----------
    face_count : int
      Number of faces desired in mesh

    Returns
    -----------
    soup : trimesh.Trimesh
      Geometry with face_count random faces
    """
    vertices = np.random.random((face_count * 3, 3)) - 0.5
    faces = np.arange(face_count * 3).reshape((-1, 3))
    soup = Trimesh(vertices=vertices, faces=faces)
    return soup


def axis(origin_size=0.04,
         transform=None,
         origin_color=None,
         axis_radius=None,
         axis_length=None):
    """
    Return an XYZ axis marker as a  Trimesh, which represents position
    and orientation. If you set the origin size the other parameters
    will be set relative to it.

    Parameters
    ----------
    transform : (4, 4) float
      Transformation matrix
    origin_size : float
      Radius of sphere that represents the origin
    origin_color : (3,) float or int, uint8 or float
      Color of the origin
    axis_radius : float
      Radius of cylinder that represents x, y, z axis
    axis_length: float
      Length of cylinder that represents x, y, z axis

    Returns
    -------
    marker : trimesh.Trimesh
      Mesh geometry of axis indicators
    """
    # the size of the ball representing the origin
    origin_size = float(origin_size)

    # set the transform and use origin-relative
    # sized for other parameters if not specified
    if transform is None:
        transform = np.eye(4)
    if origin_color is None:
        origin_color = [255, 255, 255, 255]
    if axis_radius is None:
        axis_radius = origin_size / 5.0
    if axis_length is None:
        axis_length = origin_size * 10.0

    # generate a ball for the origin
    axis_origin = uv_sphere(radius=origin_size,
                            count=[10, 10])
    axis_origin.apply_transform(transform)

    # apply color to the origin ball
    axis_origin.visual.face_colors = origin_color

    # create the cylinder for the z-axis
    translation = transformations.translation_matrix(
        [0, 0, axis_length / 2])
    z_axis = cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(translation))
    # XYZ->RGB, Z is blue
    z_axis.visual.face_colors = [0, 0, 255]

    # create the cylinder for the y-axis
    translation = transformations.translation_matrix(
        [0, 0, axis_length / 2])
    rotation = transformations.rotation_matrix(np.radians(-90),
                                               [1, 0, 0])
    y_axis = cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(rotation).dot(translation))
    # XYZ->RGB, Y is green
    y_axis.visual.face_colors = [0, 255, 0]

    # create the cylinder for the x-axis
    translation = transformations.translation_matrix(
        [0, 0, axis_length / 2])
    rotation = transformations.rotation_matrix(np.radians(90),
                                               [0, 1, 0])
    x_axis = cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(rotation).dot(translation))
    # XYZ->RGB, X is red
    x_axis.visual.face_colors = [255, 0, 0]

    # append the sphere and three cylinders
    marker = util.concatenate([axis_origin,
                               x_axis,
                               y_axis,
                               z_axis])
    return marker


def camera_marker(camera,
                  marker_height=0.4,
                  origin_size=None):
    """
    Create a visual marker for a camera object, including an axis and FOV.

    Parameters
    ---------------
    camera : trimesh.scene.Camera
      Camera object with FOV and transform defined
    marker_height : float
      How far along the camera Z should FOV indicators be
    origin_size : float
      Sphere radius of the origin (default: marker_height / 10.0)

    Returns
    ------------
    meshes : list
      Contains Trimesh and Path3D objects which can be visualized
    """

    camera_transform = camera.transform
    if camera_transform is None:
        camera_transform = np.eye(4)

    # append the visualizations to an array
    meshes = [axis(origin_size=marker_height / 10.0)]
    meshes[0].apply_transform(camera_transform)

    try:
        # path is a soft dependency
        from .path.exchange.load import load_path
    except ImportError:
        # they probably don't have shapely installed
        log.warning('unable to create FOV visualization!',
                    exc_info=True)
        return meshes

    # create sane origin size from marker height
    if origin_size is None:
        origin_size = marker_height / 10.0

    # calculate vertices from camera FOV angles
    x = marker_height * np.tan(np.deg2rad(camera.fov[0]) / 2.0)
    y = marker_height * np.tan(np.deg2rad(camera.fov[1]) / 2.0)
    z = marker_height

    # combine the points into the vertices of an FOV visualization
    points = transformations.transform_points(
        [(0, 0, 0),
         (-x, -y, z),
         (x, -y, z),
         (x, y, z),
         (-x, y, z)],
        matrix=camera_transform)

    # create line segments for the FOV visualization
    # a segment from the origin to each bound of the FOV
    segments = np.column_stack(
        (np.zeros_like(points), points)).reshape(
        (-1, 3))

    # add a loop for the outside of the FOV then reshape
    # the whole thing into multiple line segments
    segments = np.vstack((segments,
                          points[[1, 2,
                                  2, 3,
                                  3, 4,
                                  4, 1]])).reshape((-1, 2, 3))

    # add a single Path3D object for all line segments
    meshes.append(load_path(segments))

    return meshes
