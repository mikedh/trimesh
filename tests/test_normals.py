try:
    from . import generic as g
except BaseException:
    import generic as g


class NormalsTest(g.unittest.TestCase):

    def test_vertex_normal(self):
        mesh = g.trimesh.creation.icosahedron()
        # the icosahedron is centered at zero, so the true vertex
        # normal is just a unit vector of the vertex position
        truth = g.trimesh.util.unitize(mesh.vertices)

        # force fallback to loop normal summing by passing None
        # as the sparse matrix
        normals = g.trimesh.geometry.mean_vertex_normals(
            len(mesh.vertices),
            mesh.faces,
            mesh.face_normals,
            sparse=None)
        assert g.np.allclose(normals - truth, 0.0)

        # make sure the automatic sparse matrix generation works
        normals = g.trimesh.geometry.mean_vertex_normals(
            len(mesh.vertices),
            mesh.faces,
            mesh.face_normals)
        assert g.np.allclose(normals - truth, 0.0)

        # make sure the Trimesh normals- related attributes
        # are wired correctly
        assert mesh.faces_sparse is not None
        assert mesh.vertex_normals.shape == mesh.vertices.shape
        assert g.np.allclose(mesh.vertex_normals - truth, 0.0)

    def test_face_normals(self):
        """
        Test automatic generation of face normals on mesh objects
        """
        mesh = g.trimesh.creation.icosahedron()
        assert mesh.face_normals.shape == mesh.faces.shape

        # normals should regenerate
        mesh.face_normals = None
        assert mesh.face_normals.shape == mesh.faces.shape

        # we shouldn't be able to assign stupid wrong values
        # even with nonzero and the right shape
        mesh.face_normals = g.np.ones_like(mesh.faces) * [0.0, 0.0, 1.0]
        assert not g.np.allclose(mesh.face_normals, [0.0, 0.0, 1.0])

        # setting normals to None should force recompute
        mesh.face_normals = None
        assert mesh.face_normals is not None
        assert not g.np.allclose(mesh.face_normals,
                                 [0.0, 0.0, 1.0])

        # setting face normals to zeros shouldn't work
        mesh.face_normals = g.np.zeros_like(mesh.faces)
        assert g.np.allclose(
            g.np.linalg.norm(mesh.face_normals, axis=1), 1.0)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
