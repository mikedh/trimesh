try:
    from . import generic as g
except BaseException:
    import generic as g


class SubDivideTest(g.unittest.TestCase):
    def test_subdivide(self):
        meshes = [
            g.get_mesh("soup.stl"),  # a soup of random triangles
            g.get_mesh("cycloidal.ply"),  # a mesh with multiple bodies
            g.get_mesh("featuretype.STL"),
        ]  # a mesh with a single body

        for m in meshes:
            sub = m.subdivide()
            assert g.np.allclose(m.area, sub.area)
            assert len(sub.faces) > len(m.faces)

            max_edge = m.scale / 50
            sub, idx = m.subdivide_to_size(max_edge=max_edge, return_index=True)
            assert g.np.allclose(m.area, sub.area)
            edge_len = (
                g.np.diff(sub.vertices[sub.edges_unique], axis=1).reshape((-1, 3)) ** 2
            ).sum(axis=1) ** 0.5
            assert (edge_len < max_edge).all()

            # should be the same order of magnitude size
            assert g.np.allclose(m.extents, sub.extents, rtol=2)
            # should be one index per new face
            assert len(idx) == len(sub.faces)
            # every face should be subdivided
            assert idx.max() == (len(m.faces) - 1)

            # check the original face index using barycentric coordinates
            epsilon = 1e-3
            for vid in sub.faces.T:
                # find the barycentric coordinates
                bary = g.trimesh.triangles.points_to_barycentric(
                    m.triangles[idx], sub.vertices[vid]
                )
                # if face indexes are correct they will be on the triangle
                # which means all barycentric coordinates are between 0.0-1.0
                assert bary.max() < (1 + epsilon)
                assert bary.min() > -epsilon
                # make sure it's not all zeros
                assert bary.ptp() > epsilon

            v, f = g.trimesh.remesh.subdivide(vertices=m.vertices, faces=m.faces)

            max_edge = m.scale / 50
            v, f, idx = g.trimesh.remesh.subdivide_to_size(
                vertices=m.vertices, faces=m.faces, max_edge=max_edge, return_index=True
            )
            ms = g.trimesh.Trimesh(vertices=v, faces=f)
            assert g.np.allclose(m.area, ms.area)
            edge_len = (
                g.np.diff(ms.vertices[ms.edges_unique], axis=1).reshape((-1, 3)) ** 2
            ).sum(axis=1) ** 0.5
            assert (edge_len < max_edge).all()

            # should be one index per new face
            assert len(idx) == len(f)
            # every face should be subdivided
            assert idx.max() == (len(m.faces) - 1)

            # check the original face index using barycentric coordinates
            epsilon = 1e-3
            for vid in f.T:
                # find the barycentric coordinates
                bary = g.trimesh.triangles.points_to_barycentric(m.triangles[idx], v[vid])
                # if face indexes are correct they will be on the triangle
                # which means all barycentric coordinates are between 0.0-1.0
                assert bary.max() < (1 + epsilon)
                assert bary.min() > -epsilon
                # make sure it's not all zeros
                assert bary.ptp() > epsilon

            check = m.subdivide_to_size(
                max_edge=m.extents.sum(), max_iter=1, return_index=False
            )
            assert check.faces.shape == m.faces.shape

    def test_sub(self):
        # try on some primitives
        meshes = [g.trimesh.creation.box(), g.trimesh.creation.icosphere()]

        for m in meshes:
            s = m.subdivide(face_index=[0, len(m.faces) - 1])
            # shouldn't have subdivided in-place
            assert len(s.faces) > len(m.faces)
            # area should be the same
            assert g.np.isclose(m.area, s.area)
            # volume should be the same
            assert g.np.isclose(m.volume, s.volume)

            max_edge = m.scale / 50
            s = m.subdivide_to_size(max_edge=max_edge)
            # shouldn't have subdivided in-place
            assert len(s.faces) > len(m.faces)
            # area should be the same
            assert g.np.isclose(m.area, s.area)
            # volume should be the same
            assert g.np.isclose(m.volume, s.volume)

    def test_loop(self):
        meshes = [
            g.get_mesh("soup.stl"),  # a soup of random triangles
            g.get_mesh("featuretype.STL"),
        ]  # a mesh with a single body

        for m in meshes:
            sub = m.subdivide_loop(iterations=1)
            # number of faces should increase
            assert len(sub.faces) > len(m.faces)
            # subdivided faces are smaller
            assert sub.area_faces.mean() < m.area_faces.mean()

    def test_loop_multibody(self):
        # a mesh with multiple bodies
        mesh = g.get_mesh("cycloidal.ply")
        sub = mesh.subdivide_loop(iterations=2)

        # number of faces should increase
        assert len(sub.faces) > len(mesh.faces)
        # subdivided faces are smaller
        assert sub.area_faces.mean() < mesh.area_faces.mean()
        # should be the same order of magnitude area
        # rtol=2 means it can be up to twice/half
        assert g.np.isclose(sub.area, mesh.area, rtol=2)
        # should have the same number of bodies
        assert len(mesh.split()) == len(sub.split())

    def test_loop_multi_simple(self, count=10):
        meshes = []
        for i in range(count):
            current = g.trimesh.creation.icosahedron()
            current.apply_translation([i * 1.5, 0, 0])
            meshes.append(current)
        # concatenate into a single multibody mesh
        m = g.trimesh.util.concatenate(meshes)
        # run subdivision on that
        a = m.subdivide_loop(iterations=4)
        # make sure it splits and is watertight
        split = a.split()
        assert len(split) == count
        assert all(i.is_watertight for i in split)

    def test_loop_correct(self):
        box = g.trimesh.creation.box()
        big_sphere = g.trimesh.creation.icosphere(radius=0.5)
        small_sphere = g.trimesh.creation.icosphere(radius=0.4)
        sub = box.subdivide_loop(iterations=2)
        # smaller than 0.5 sphere
        assert big_sphere.contains(sub.vertices).all()
        # bigger than 0.4 sphere
        assert (~small_sphere.contains(sub.vertices)).all()

    def test_loop_bound(self):
        def _get_boundary_vertices(mesh):
            boundary_groups = g.trimesh.grouping.group_rows(
                mesh.edges_sorted, require_count=1
            )
            return mesh.vertices[g.np.unique(mesh.edges_sorted[boundary_groups])]

        box = g.trimesh.creation.box()
        bottom_mask = g.np.zeros(len(box.faces), dtype=bool)
        bottom_faces = [1, 5]
        bottom_mask[bottom_faces] = True
        # eliminate bottom of the box
        box.update_faces(~bottom_mask)
        bottom_vrts = _get_boundary_vertices(box)
        # subdivide box
        sub = box.subdivide_loop(iterations=2)
        sub_bottom_vrts = _get_boundary_vertices(sub)

        # y value of bottom boundary vertices should not be changed
        assert g.np.isclose(
            bottom_vrts[:, 1].mean(), sub_bottom_vrts[:, 1].mean(), atol=1e-5
        )

    def test_uv(self):
        # get a mesh with texture
        m = g.get_mesh("fuze.obj")
        # m.show()
        # get the shape of the initial mesh
        shape = m.vertices.shape
        # subdivide the mesh
        s = m.subdivide()

        # shouldn't have changed source mesh
        assert m.vertices.shape == shape
        # subdivided mesh should have more vertices
        assert s.vertices.shape[0] > shape[0]
        # should have UV coordinates matching vertices
        assert s.vertices.shape[0] == s.visual.uv.shape[0]

        # original UV coordinates with faces
        ov = m.visual.uv[m.faces]
        # subdivided mesh faces
        sv = s.visual.uv[s.faces]
        # both subdivided and original should have faces
        # that don't vary wildly
        assert ov.ptp(axis=1).mean(axis=0).max() < 0.1
        assert sv.ptp(axis=1).mean(axis=0).max() < 0.1

        max_edge = m.scale / 50
        s = m.subdivide_to_size(max_edge=max_edge)

        # shouldn't have changed source mesh
        assert m.vertices.shape == shape
        # subdivided mesh should have more vertices
        assert s.vertices.shape[0] > shape[0]
        # should have UV coordinates matching vertices
        assert s.vertices.shape[0] == s.visual.uv.shape[0]

        # original UV coordinates with faces
        ov = m.visual.uv[m.faces]
        # subdivided mesh faces
        sv = s.visual.uv[s.faces]
        # both subdivided and original should have faces
        # that don't vary wildly
        assert ov.ptp(axis=1).mean(axis=0).max() < 0.1
        assert sv.ptp(axis=1).mean(axis=0).max() < 0.1

    def test_max_iter(self):
        m = g.trimesh.creation.box()
        try:
            m.subdivide_to_size(0.01, max_iter=1)
            raise BaseException("that shouldn't have worked!")
        except ValueError:
            # this shouldn't have worked
            pass
        # this should be enough iterations
        r = m.subdivide_to_size(0.01, max_iter=10)
        assert r.is_watertight

    def test_idx_simple(self):
        vertices = g.np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0]).reshape((-1, 3)) * 10
        faces = g.np.array(
            [
                0,
                1,
                2,
                0,
                2,
                3,
            ]
        ).reshape((-1, 3))

        def test(fidx):
            v, f, idx = g.trimesh.remesh.subdivide(
                vertices, faces, face_index=fidx, return_index=True
            )
            eps = 1e-8
            for fid in fidx:
                # get the new triangles, as indicated by the index
                tri_new = v[f[idx[fid]]]

                # this is the original triangle
                original = vertices[faces[g.np.tile(fid, len(tri_new) * 3)]]

                bary = g.trimesh.triangles.points_to_barycentric(
                    triangles=original, points=tri_new.reshape((-1, 3))
                )

                assert (bary < 1 + eps).all()
                assert (bary > -eps).all()

        test([0, 1])
        test([1, 0])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
