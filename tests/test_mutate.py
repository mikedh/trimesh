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
    assert not verts.flags["WRITEABLE"]
    assert not faces.flags["WRITEABLE"]

    mesh = g.trimesh.Trimesh(verts, faces, process=False, validate=False)
    assert not mesh.vertices.flags["WRITEABLE"]
    assert not mesh.faces.flags["WRITEABLE"]

    # return the mesh, and read-only vertices and faces
    return mesh, verts, faces


class MutateTests(g.unittest.TestCase):
    def test_not_mutated_cube(self):
        self._test_not_mutated(*get_readonly("cube.OBJ"))

    def test_not_mutated_torus(self):
        self._test_not_mutated(*get_readonly("torus.STL"))

    def test_not_mutated_bunny(self):
        self._test_not_mutated(*get_readonly("bunny.ply"))

    def test_not_mutated_teapot(self):
        self._test_not_mutated(*get_readonly("teapot.stl"))

    def _test_not_mutated(self, mesh, verts, faces):
        verts = g.np.copy(verts)
        faces = g.np.copy(faces)
        lo, hi = mesh.bounds

        _ = mesh.faces_sparse
        _ = mesh.face_normals
        _ = mesh.vertex_normals
        _ = mesh.extents
        _ = mesh.scale
        _ = mesh.centroid
        _ = mesh.center_mass
        _ = mesh.density
        _ = mesh.volume
        _ = mesh.mass
        _ = mesh.moment_inertia
        _ = mesh.principal_inertia_components
        _ = mesh.principal_inertia_vectors
        _ = mesh.principal_inertia_transform
        _ = mesh.symmetry
        _ = mesh.symmetry_axis
        _ = mesh.symmetry_section
        _ = mesh.triangles
        _ = mesh.triangles_tree
        _ = mesh.triangles_center
        _ = mesh.triangles_cross
        _ = mesh.edges
        _ = mesh.edges_face
        _ = mesh.edges_unique
        _ = mesh.edges_unique_length
        _ = mesh.edges_unique_inverse
        _ = mesh.edges_sorted
        _ = mesh.edges_sparse
        _ = mesh.body_count
        _ = mesh.faces_unique_edges
        _ = mesh.euler_number
        _ = mesh.referenced_vertices
        _ = mesh.units
        _ = mesh.face_adjacency
        _ = mesh.face_adjacency_edges
        _ = mesh.face_adjacency_angles
        _ = mesh.face_adjacency_projections
        _ = mesh.face_adjacency_convex
        _ = mesh.face_adjacency_unshared
        _ = mesh.face_adjacency_radius
        _ = mesh.face_adjacency_span
        _ = mesh.vertex_adjacency_graph
        _ = mesh.vertex_neighbors
        _ = mesh.is_winding_consistent
        _ = mesh.is_watertight
        _ = mesh.is_volume
        _ = mesh.is_empty
        _ = mesh.is_convex
        _ = mesh.kdtree
        _ = mesh.facets
        _ = mesh.facets_area
        _ = mesh.facets_normal
        _ = mesh.facets_origin
        _ = mesh.facets_boundary
        _ = mesh.facets_on_hull
        _ = mesh.visual
        _ = mesh.convex_hull
        _ = mesh.sample(500, False)
        _ = mesh.voxelized((hi[0] - lo[0]) / 100.0)
        _ = mesh.outline()
        _ = mesh.area
        _ = mesh.area_faces
        _ = mesh.mass_properties
        _ = mesh.scene()
        _ = mesh.identifier
        _ = mesh.identifier_hash
        _ = mesh.to_dict()
        _ = mesh.face_angles
        _ = mesh.face_angles_sparse
        _ = mesh.vertex_defects
        _ = mesh.face_adjacency_tree
        _ = mesh.copy()

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

        assert not mesh.vertices.flags["WRITEABLE"]
        assert not mesh.faces.flags["WRITEABLE"]
        assert g.np.all(g.np.isclose(verts, mesh.vertices))
        assert g.np.all(g.np.isclose(faces, mesh.faces))


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
