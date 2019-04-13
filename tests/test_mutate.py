try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np

import trimesh


def readonly_mesh(model):
    mesh  = g.get_mesh(model)
    verts = mesh.vertices
    faces = mesh.faces
    verts = np.ndarray(verts.shape, verts.dtype, bytes(verts.tostring()))
    faces = np.ndarray(faces.shape, faces.dtype, bytes(faces.tostring()))
    return (trimesh.Trimesh(verts, faces, process=False, validate=False),
            verts,
            faces)


class MutateTests(g.unittest.TestCase):

    def test_not_mutated_cube(self):
        self._test_not_mutated(*readonly_mesh('cube.OBJ'))

    def test_not_mutated_torus(self):
        self._test_not_mutated(*readonly_mesh('torus.STL'))

    def test_not_mutated_bunny(self):
        self._test_not_mutated(*readonly_mesh('bunny.ply'))

    def test_not_mutated_teapot(self):
        self._test_not_mutated(*readonly_mesh('teapot.stl'))

    def _test_not_mutated(self, mesh, verts, faces):
        verts  = np.copy(verts)
        faces  = np.copy(faces)
        lo, hi = mesh.bounds

        mesh.faces_sparse
        mesh.face_normals
        mesh.vertex_normals
        mesh.extents
        mesh.scale
        mesh.centroid
        mesh.center_mass
        mesh.density
        mesh.volume
        mesh.mass
        mesh.moment_inertia
        mesh.principal_inertia_components
        mesh.principal_inertia_vectors
        mesh.principal_inertia_transform
        mesh.symmetry
        mesh.symmetry_axis
        mesh.symmetry_section
        mesh.triangles
        mesh.triangles_tree
        mesh.triangles_center
        mesh.triangles_cross
        mesh.edges
        mesh.edges_face
        mesh.edges_unique
        mesh.edges_unique_length
        mesh.edges_unique_inverse
        mesh.edges_sorted
        mesh.edges_sparse
        mesh.body_count
        mesh.faces_unique_edges
        mesh.euler_number
        mesh.referenced_vertices
        mesh.units
        mesh.face_adjacency
        mesh.face_adjacency_edges
        mesh.face_adjacency_angles
        mesh.face_adjacency_projections
        mesh.face_adjacency_convex
        mesh.face_adjacency_unshared
        mesh.face_adjacency_radius
        mesh.face_adjacency_span
        mesh.vertex_adjacency_graph
        mesh.vertex_neighbors
        mesh.is_winding_consistent
        mesh.is_watertight
        mesh.is_volume
        mesh.is_empty
        mesh.is_convex
        mesh.kdtree
        mesh.facets
        mesh.facets_area
        mesh.facets_normal
        mesh.facets_origin
        mesh.facets_boundary
        mesh.facets_on_hull
        mesh.visual
        mesh.convex_hull
        mesh.sample(500, False)
        mesh.voxelized((hi[0] - lo[0]) / 100.0)
        mesh.outline()
        mesh.area
        mesh.area_faces
        mesh.mass_properties
        mesh.scene()
        mesh.identifier
        mesh.identifier_md5
        mesh.to_dict()
        mesh.face_angles
        mesh.face_angles_sparse
        mesh.vertex_defects
        mesh.face_adjacency_tree
        mesh.copy()

        # ray.intersects_id
        centre              = mesh.vertices.mean(axis=0)
        origins             = np.random.random((100, 3)) * 1000
        directions          = np.copy(origins)
        directions[:50,  :] -= centre
        directions[ 50:, :] += centre
        mesh.ray.intersects_id(origins, directions)

        # nearest.vertex
        points = np.random.random((500, 3)) * 100
        mesh.nearest.vertex(points)

        # section
        origins = np.random.random((500, 3)) * 100
        normals = np.random.random((500, 3)) * 100
        heights = np.random.random((50,))    * 100
        for o, n in zip(origins, normals):
            mesh.slice_plane(o, n)
            mesh.section(o, n)
            mesh.section_multiplane(o, n, heights)

        assert not mesh.vertices.flags.writeable
        assert not mesh.faces   .flags.writeable
        assert np.all(np.isclose(verts, mesh.vertices))
        assert np.all(np.isclose(faces, mesh.faces))
