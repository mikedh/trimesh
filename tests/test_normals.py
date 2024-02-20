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
            len(mesh.vertices), mesh.faces, mesh.face_normals
        )
        assert g.np.allclose(normals - truth, 0.0)

        # make sure the automatic sparse matrix generation works
        normals = g.trimesh.geometry.mean_vertex_normals(
            len(mesh.vertices), mesh.faces, mesh.face_normals
        )
        assert g.np.allclose(normals - truth, 0.0)

        # make sure the Trimesh normals- related attributes
        # are wired correctly
        assert mesh.faces_sparse is not None
        assert mesh.vertex_normals.shape == mesh.vertices.shape
        assert g.np.allclose(mesh.vertex_normals - truth, 0.0)

    def test_weighted_vertex_normals(self):
        def compare_trimesh_to_groundtruth(mesh, truth, atol=g.trimesh.tol.merge):
            # force fallback to loop normal summing by passing None
            # as the sparse matrix
            normals = g.trimesh.geometry.weighted_vertex_normals(
                vertex_count=len(mesh.vertices),
                faces=mesh.faces,
                face_normals=mesh.face_normals,
                face_angles=mesh.face_angles,
            )
            assert g.np.allclose(normals, truth, atol=atol)

            # make sure the automatic sparse matrix generation works
            normals = g.trimesh.geometry.weighted_vertex_normals(
                len(mesh.vertices), mesh.faces, mesh.face_normals, mesh.face_angles
            )
            assert g.np.allclose(normals - truth, 0.0, atol=atol)

            # make sure the Trimesh normals- related attributes
            # are wired correctly
            assert mesh.faces_sparse is not None
            assert mesh.vertex_normals.shape == mesh.vertices.shape
            assert g.np.allclose(mesh.vertex_normals - truth, 0.0, atol=atol)

        # the icosahedron is centered at zero, so the true vertex
        # normal is just a unit vector of the vertex position
        ico_mesh = g.trimesh.creation.icosahedron()
        ico_truth = g.trimesh.util.unitize(ico_mesh.vertices)
        compare_trimesh_to_groundtruth(ico_mesh, ico_truth)

        # create a cube centered at zero, as with the icosahedron,
        # normals compute as unit vectors of the corner vertices
        # due to the triangulation of the box, this case would fail
        # with a simple face-normals-average as vertex-normal method
        box_mesh = g.trimesh.creation.box()
        box_truth = g.trimesh.util.unitize(box_mesh.vertices)
        compare_trimesh_to_groundtruth(box_mesh, box_truth)

        # load the fandisk model: a complex triangulated mesh with
        # smooth curved surfaces, corners and sharp creases
        # ground truth vertex normals were computed in MeshLab and
        # are included in the file
        fandisk_mesh = g.get_mesh("fandisk.obj")
        fandisk_truth = fandisk_mesh.vertex_normals
        # due to the limited precision in the MeshLab export,
        # we have to tweak the tolerance for the comparison a little
        compare_trimesh_to_groundtruth(fandisk_mesh, fandisk_truth, 0.0001)

        # see how we do with degenerate faces
        m = g.trimesh.creation.box()
        m.faces[0][0] = m.faces[0][1]
        norm = m.vertex_normals
        assert g.np.isfinite(norm).all()
        assert len(norm) == len(m.vertices)

        # vertices with every face intact
        mask = g.np.zeros(len(m.vertices), dtype=bool)
        mask[m.faces[0]] = False
        # it's a box so normals should all be unit vectors [1,1,1]
        assert g.np.allclose(g.np.abs(norm[mask]), (1.0 / 3.0) ** 0.5)

        # try with a deliberately broken sparse matrix to test looping path
        norm = g.trimesh.geometry.weighted_vertex_normals(
            vertex_count=len(m.vertices),
            faces=m.faces,
            face_normals=m.face_normals,
            face_angles=m.face_angles,
            use_loop=True,
        )
        assert g.np.isfinite(norm).all()
        assert len(norm) == len(m.vertices)

        # every intact vertex should be away from box corner
        assert g.np.allclose(g.np.abs(norm[mask]), (1.0 / 3.0) ** 0.5)

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
        assert not g.np.allclose(mesh.face_normals, [0.0, 0.0, 1.0])

        # setting face normals to zeros shouldn't work
        mesh.face_normals = g.np.zeros_like(mesh.faces)
        assert g.np.allclose(g.np.linalg.norm(mesh.face_normals, axis=1), 1.0)

    def test_merge(self):
        """
        Check merging with vertex normals
        """
        # no vertex merging
        m = g.get_mesh("cube_compressed.obj", process=False)
        assert m.vertices.shape == (24, 3)
        assert m.faces.shape == (12, 3)
        assert g.np.isclose(m.volume, 8.0, atol=1e-4)

        # with normal-aware vertex merging
        m = g.get_mesh("cube_compressed.obj", process=True)
        assert m.vertices.shape == (24, 3)
        assert m.faces.shape == (12, 3)
        assert g.np.isclose(m.volume, 8.0, atol=1e-4)

        # without considering normals should just be cube
        m.merge_vertices(merge_norm=True)
        assert m.vertices.shape == (8, 3)
        assert m.faces.shape == (12, 3)
        assert g.np.isclose(m.volume, 8.0, atol=1e-4)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
