"""
test_mutate.py
---------------

Make sure trimesh operations aren't mutating things
that they shouldn't be.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g


def get_readonly(model_name):
    """
    Get a mesh and make vertices and faces read only.

    Parameters
    ------------
    model_name : str
     Model name in models directory

    Returns
    -----------
    mesh : trimesh.Trimesh
      Geometry with read-only data
    verts : (n, 3) float
      Read- only vertices
    faces : (m, 3) int
      Read- only faces
    """
    original = g.get_mesh(model_name)
    # get the original data from the mesh
    verts = original.vertices
    faces = original.faces
    # use the buffer interface to generate read-only arrays
    verts = g.np.ndarray(verts.shape, verts.dtype, bytes(verts.tobytes()))
    faces = g.np.ndarray(faces.shape, faces.dtype, bytes(faces.tobytes()))
    # everything should be read only now
    assert not verts.flags['WRITEABLE']
    assert not faces.flags['WRITEABLE']

    mesh = g.trimesh.Trimesh(verts, faces, process=False, validate=False)
    assert not mesh.vertices.flags['WRITEABLE']
    assert not mesh.faces.flags['WRITEABLE']

    # return the mesh, and read-only vertices and faces
    return mesh, verts, faces


class MutateTests(g.unittest.TestCase):

    def test_not_mutated_cube(self):
        self._test_not_mutated(*get_readonly('cube.OBJ'))

    def test_not_mutated_torus(self):
        self._test_not_mutated(*get_readonly('torus.STL'))

    def test_not_mutated_bunny(self):
        self._test_not_mutated(*get_readonly('bunny.ply'))

    def test_not_mutated_teapot(self):
        self._test_not_mutated(*get_readonly('teapot.stl'))

    def _test_not_mutated(self, mesh, verts, faces):
        verts = g.np.copy(verts)
        faces = g.np.copy(faces)
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
        mesh.identifier_hash
        mesh.to_dict()
        mesh.face_angles
        mesh.face_angles_sparse
        mesh.vertex_defects
        mesh.face_adjacency_tree
        mesh.copy()

        # ray.intersects_id
        centre = mesh.vertices.mean(axis=0)
        origins = g.random((100, 3)) * 1000
        directions = g.np.copy(origins)
        directions[:50, :] -= centre
        directions[50:, :] += centre
        mesh.ray.intersects_id(origins, directions)

        # nearest.vertex
        points = g.random((500, 3)) * 100
        mesh.nearest.vertex(points)

        # section
        section_count = 20
        origins = g.random((section_count, 3)) * 100
        normals = g.random((section_count, 3)) * 100
        heights = g.random((10,)) * 100
        for o, n in zip(origins, normals):
            # try slicing at random origin and at center mass
            mesh.slice_plane(o, n)
            mesh.slice_plane(mesh.center_mass, n)

            # section at random origin and center mass
            mesh.section(o, n)
            mesh.section(mesh.center_mass, n)

            # same with multiplane
            mesh.section_multiplane(o, n, heights)
            mesh.section_multiplane(mesh.center_mass, n, heights)

        assert not mesh.vertices.flags['WRITEABLE']
        assert not mesh.faces.flags['WRITEABLE']
        assert g.np.all(g.np.isclose(verts, mesh.vertices))
        assert g.np.all(g.np.isclose(faces, mesh.faces))


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
