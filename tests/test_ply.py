try:
    from . import generic as g
except BaseException:
    import generic as g


class PlyTest(g.unittest.TestCase):
    def test_ply_dtype(self):
        # make sure all ply dtype strings are valid dtypes
        dtypes = g.trimesh.exchange.ply._dtypes
        for d in dtypes.values():
            # will raise if dtype string not valid
            g.np.dtype(d)

    def test_ply(self):
        m = g.get_mesh("machinist.XAML")

        assert m.visual.kind == "face"
        assert m.visual.face_colors.ptp(axis=0).max() > 0

        export = m.export(file_type="ply")
        reconstructed = g.wrapload(export, file_type="ply")

        assert reconstructed.visual.kind == "face"

        assert g.np.allclose(reconstructed.visual.face_colors, m.visual.face_colors)

        m = g.get_mesh("reference.ply")

        assert m.visual.kind == "vertex"
        assert m.visual.vertex_colors.ptp(axis=0).max() > 0

        export = m.export(file_type="ply")
        reconstructed = g.wrapload(export, file_type="ply")
        assert reconstructed.visual.kind == "vertex"

        assert g.np.allclose(reconstructed.visual.vertex_colors, m.visual.vertex_colors)

    def test_points(self):
        # Test reading point clouds from PLY files

        m = g.get_mesh("points_ascii.ply")
        assert isinstance(m, g.trimesh.PointCloud)
        assert m.vertices.shape == (5, 3)

        m = g.get_mesh("points_bin.ply")
        assert m.vertices.shape == (5, 3)
        assert isinstance(m, g.trimesh.PointCloud)

        m = g.get_mesh("points_emptyface.ply")
        assert m.vertices.shape == (1024, 3)
        assert isinstance(m, g.trimesh.PointCloud)

    def test_list_properties(self):
        """
        Test reading point clouds with the following metadata:
        - lists of differing length
        - multiple list properties
        - single-element properties that come after list properties
        """
        m = g.get_mesh("points_ascii_with_lists.ply")

        point_list = m.metadata["_ply_raw"]["point_list"]["data"]
        assert g.np.array_equal(
            point_list["point_indices1"][0], g.np.array([10, 11, 12], dtype=g.np.uint32)
        )
        assert g.np.array_equal(
            point_list["point_indices1"][1], g.np.array([10, 11], dtype=g.np.uint32)
        )
        assert g.np.array_equal(
            point_list["point_indices2"][0], g.np.array([13, 14], dtype=g.np.uint32)
        )
        assert g.np.array_equal(
            point_list["point_indices2"][1], g.np.array([12, 13, 14], dtype=g.np.uint32)
        )
        assert g.np.array_equal(
            point_list["some_float"], g.np.array([1.1, 2.2], dtype=g.np.float32)
        )

    def test_vertex_attributes(self):
        """
        Test writing vertex attributes to a ply, by reading them back and asserting the
        written attributes array matches
        """

        m = g.get_mesh("box.STL")
        test_1d_attribute = g.np.copy(m.vertices[:, 0])
        test_nd_attribute = g.np.copy(m.vertices)
        m.vertex_attributes["test_1d_attribute"] = test_1d_attribute
        m.vertex_attributes["test_nd_attribute"] = test_nd_attribute

        export = m.export(file_type="ply")
        reconstructed = g.wrapload(export, file_type="ply")

        vertex_attributes = reconstructed.metadata["_ply_raw"]["vertex"]["data"]
        result_1d = vertex_attributes["test_1d_attribute"]
        result_nd = vertex_attributes["test_nd_attribute"]["f1"]

        g.np.testing.assert_almost_equal(result_1d, test_1d_attribute)
        g.np.testing.assert_almost_equal(result_nd, test_nd_attribute)

    def test_face_attributes(self):
        # Test writing face attributes to a ply, by reading
        # them back and asserting the written attributes array matches

        m = g.get_mesh("box.STL")
        test_1d_attribute = g.np.copy(m.face_angles[:, 0])
        test_nd_attribute = g.np.copy(m.face_angles)
        m.face_attributes["test_1d_attribute"] = test_1d_attribute
        m.face_attributes["test_nd_attribute"] = test_nd_attribute

        export = m.export(file_type="ply")
        reconstructed = g.wrapload(export, file_type="ply")

        face_attributes = reconstructed.metadata["_ply_raw"]["face"]["data"]
        result_1d = face_attributes["test_1d_attribute"]
        result_nd = face_attributes["test_nd_attribute"]["f1"]

        g.np.testing.assert_almost_equal(result_1d, test_1d_attribute)
        g.np.testing.assert_almost_equal(result_nd, test_nd_attribute)

        no_attr = m.export(file_type="ply", include_attributes=False)
        assert len(no_attr) < len(export)

    def test_cases(self):
        a = g.get_mesh("featuretype.STL")
        b = g.get_mesh("featuretype.ply")
        assert a.faces.shape == b.faces.shape

        # has mixed quads and triangles
        m = g.get_mesh("suzanne.ply")
        assert len(m.faces) > 0

    def test_ascii_color(self):
        mesh = g.trimesh.creation.box()
        en = g.wrapload(mesh.export(file_type="ply", encoding="ascii"), file_type="ply")
        assert en.visual.kind is None

        color = [255, 0, 0, 255]
        mesh.visual.vertex_colors = color

        # try exporting and reloading raw
        eb = g.wrapload(mesh.export(file_type="ply"), file_type="ply")

        assert g.np.allclose(eb.visual.vertex_colors[0], color)
        assert eb.visual.kind == "vertex"

        ea = g.wrapload(mesh.export(file_type="ply", encoding="ascii"), file_type="ply")
        assert g.np.allclose(ea.visual.vertex_colors, color)
        assert ea.visual.kind == "vertex"

    def test_empty_or_pointcloud(self):
        # demo files to check
        empty_files = [
            "ply_empty_ascii.ply",
            "ply_empty_bin.ply",
            "ply_empty_header.ply",
            "ply_points_ascii.ply",
            "ply_points_bin.ply",
        ]

        for empty_file in empty_files:
            e = g.get_mesh("emptyIO/" + empty_file)
            if "empty" in empty_file:
                # result should be an empty scene
                try:
                    e.export(file_type="ply")
                except BaseException:
                    continue
                raise ValueError("should not export empty")
            elif "points" in empty_file:
                # create export
                export = e.export(file_type="ply")
                reconstructed = g.wrapload(export, file_type="ply")

                # result should be a point cloud instance
                assert isinstance(e, g.trimesh.PointCloud)
                assert hasattr(e, "vertices")
                # point cloud export should contain vertices
                assert isinstance(reconstructed, g.trimesh.PointCloud)
                assert hasattr(reconstructed, "vertices")

    def test_blender_uv(self):
        # test texture coordinate loading for Blender exported ply files
        mesh_names = []

        # test texture coordinate loading for simple triangulated
        # Blender-export
        mesh_names.append("cube_blender_uv.ply")

        # same mesh but re-exported from meshlab as binary ply (and with
        # changed header)
        mesh_names.append("cube_blender_uv_meshlab.ply")

        # test texture coordinate loading for mesh with mixed quads and
        # triangles
        mesh_names.append("suzanne.ply")

        for mesh_name in mesh_names:
            m = g.get_mesh(mesh_name)
            assert hasattr(m, "visual") and hasattr(m.visual, "uv")
            assert m.visual.uv.shape[0] == m.vertices.shape[0]

    def test_uv_export(self):
        m = g.get_mesh("fuze.ply")
        assert hasattr(m, "visual") and hasattr(m.visual, "uv")
        assert m.visual.uv.shape[0] == m.vertices.shape[0]

        # create empty file to export to

        with g.TemporaryDirectory() as D:
            name = g.os.path.join(D, "file.ply")

            # export should contain the uv data
            m.export(name)
            m2 = g.trimesh.load(name)

        assert hasattr(m2, "visual") and hasattr(m2.visual, "uv")
        assert g.np.allclose(m.visual.uv, m2.visual.uv)

    def test_fix_texture(self):
        # test loading of face indices when uv-coordinates are also contained
        m1 = g.get_mesh("plane.ply")
        m2 = g.get_mesh("plane_tri.ply")
        assert m1.faces.shape == (2, 3)
        assert m2.faces.shape == (2, 3)

    def test_texturefile(self):
        # try loading a PLY with texture
        m = g.get_mesh("fuze.ply")
        # run the checks to make sure fuze has the
        # correct number of vertices and has texture loaded
        g.check_fuze(m)

    def test_metadata(self):
        mesh = g.get_mesh("metadata.ply")

        assert (
            g.np.array([[12], [90]])
            == mesh.metadata["_ply_raw"]["face"]["data"]["face_type"]
        ).all()

    def test_point_uv(self):
        # points with UV coordinates
        # TODO shouldn't they be saved as a vertex attribute or something
        s = g.get_mesh("point_uv.ply.zip")
        p = next(iter(s.geometry.values()))
        assert p.vertices.shape == (1000, 3)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
