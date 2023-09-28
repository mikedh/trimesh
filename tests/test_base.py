"""
Test the base trimesh.Trimesh object.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class MeshTests(g.unittest.TestCase):
    def test_vertex_neighbors(self):
        m = g.trimesh.primitives.Box()
        neighbors = m.vertex_neighbors
        assert len(neighbors) == len(m.vertices)
        elist = m.edges_unique.tolist()

        for v_i, neighs in enumerate(neighbors):
            for n in neighs:
                assert [v_i, n] in elist or [n, v_i] in elist

    def test_validate(self):
        """
        Make sure meshes with validation work
        """
        m = g.get_mesh("featuretype.STL", validate=True)

        assert m.is_volume

        pre_len = len(m.vertices)
        pre_vol = m.volume

        m.remove_unreferenced_vertices()
        assert len(m.vertices) == pre_len
        assert g.np.isclose(m.volume, pre_vol)

        # add some unreferenced vertices
        m.vertices = g.np.vstack((m.vertices, g.random((100, 3))))
        assert len(m.vertices) == pre_len + 100
        assert g.np.isclose(m.volume, pre_vol)

        m.remove_unreferenced_vertices()
        assert len(m.vertices) == pre_len
        assert g.np.isclose(m.volume, pre_vol)

    def test_validate_inversion(self):
        """Make sure inverted meshes are fixed by `validate=True`"""
        orig_mesh = g.get_mesh("unit_cube.STL")
        orig_verts = orig_mesh.vertices.copy()
        orig_faces = orig_mesh.faces.copy()

        orig_face_set = {tuple(row) for row in orig_faces}

        inv_faces = orig_faces[:, ::-1]
        inv_mesh = g.Trimesh(orig_verts, inv_faces, validate=False)
        assert {tuple(row) for row in inv_mesh.faces} != orig_face_set

        fixed_mesh = g.Trimesh(orig_verts, inv_faces, validate=True)
        assert {tuple(row) for row in fixed_mesh.faces} == orig_face_set

    def test_none(self):
        """
        Make sure mesh methods don't return None or crash.
        """
        # a radially symmetric mesh with units
        # should have no properties that are None
        mesh = g.get_mesh("tube.obj")
        mesh.units = "in"

        # loop through string property names
        for method in dir(mesh):
            # ignore private- ish methods
            if method.startswith("_"):
                continue
            # a string expression to evaluate
            expr = f"mesh.{method}"

            try:
                # get the value of that expression
                res = eval(expr)
            except ImportError:
                g.log.warning("unable to import!", exc_info=True)
                continue

            # shouldn't be None!
            assert res is not None

        # check methods in scene objects
        scene = mesh.scene()
        # camera will be None unless set
        blacklist = ["camera"]
        for method in dir(scene):
            # ignore private- ish methods
            if method.startswith("_") or method in blacklist:
                continue
            # a string expression to evaluate
            expr = f"scene.{method}"
            # get the value of that expression
            res = eval(expr)
            # shouldn't be None!
            if res is None:
                raise ValueError(f'"{expr}" is None!!')


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
