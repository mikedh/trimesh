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

    def test_multi(self):
        # check to make sure we don't throw away perfectly good vertex colors
        m = g.get_mesh("multi.ply")

        assert len(m.vertex_attributes["color"]) == len(m.vertices)
        assert len(m.visual.uv) == len(m.vertices)

    def test_multi_roundtrip_preserves_uv_and_vertex_colors(self):
        # Regression test for https://github.com/mikedh/trimesh/issues/2419
        #
        # `models/multi.ply` is the fixture from the original report — a
        # textured mesh that also carries per-vertex `red green blue`.
        # The load side was fixed in 72016ad5 (constructor stores
        # `vertex_colors` as `vertex_attributes["color"]` when `visual`
        # is also passed), but the PLY exporter then wrote the array via
        # the generic vertex-attribute path as `property list uchar uchar
        # color`. That list-typed property is not standard PLY color
        # encoding and the importer (and every other PLY reader) does not
        # recover it as colors, silently losing the data on round-trip.
        m = g.get_mesh("multi.ply")
        # sanity: load preserves both
        assert m.visual.kind == "texture"
        assert m.visual.uv.shape == (len(m.vertices), 2)
        assert m.vertex_attributes["color"].shape[0] == len(m.vertices)

        # export to PLY and verify the header uses standard color properties
        export = m.export(file_type="ply")
        header = export.split(b"end_header")[0].decode("utf-8")
        assert "property uchar red" in header
        assert "property uchar green" in header
        assert "property uchar blue" in header
        assert "property uchar alpha" in header
        # texture coords still written
        assert "property double s" in header or "property float s" in header
        # the malformed list-typed `color` property must NOT appear
        assert "property list uchar uchar color" not in header
        assert "property list uchar uint8 color" not in header

        # round-trip back to a Trimesh and confirm both visual.uv AND
        # vertex_attributes["color"] survive
        reloaded = g.roundtrip(export, file_type="ply")
        assert hasattr(reloaded.visual, "uv")
        assert reloaded.visual.uv.shape == (len(reloaded.vertices), 2)
        assert "color" in reloaded.vertex_attributes
        reloaded_colors = g.np.asarray(reloaded.vertex_attributes["color"])
        assert reloaded_colors.shape == (len(reloaded.vertices), 4)
        # the RGB channels match the original (alpha is padded to 255
        # since the source PLY had only red/green/blue, which is the
        # standard PLY convention for opaque)
        orig_colors = g.np.asarray(m.vertex_attributes["color"])
        if orig_colors.shape[1] == 3:
            assert g.np.array_equal(reloaded_colors[:, :3], orig_colors)
            assert (reloaded_colors[:, 3] == 255).all()
        else:
            assert g.np.array_equal(reloaded_colors, orig_colors)

    def test_ply_export_color_only_when_rgb_shape(self):
        # Regression guard for #2419: the standard-color shortcut must
        # ONLY fire for shape-(n, 3 or 4) arrays. Non-color `color` keys
        # (e.g. a per-vertex scalar accidentally named "color", or an
        # exotic (n, 5) array) must keep falling through to the generic
        # vertex-attribute writer rather than being coerced into RGBA.
        box = g.trimesh.creation.box()
        # scalar attribute confusingly named "color" but shaped (n,)
        box.vertex_attributes["color"] = g.np.arange(
            len(box.vertices), dtype=g.np.float32
        )
        export = box.export(file_type="ply")
        header = export.split(b"end_header")[0].decode("utf-8")
        # since the array is 1-D it is exported as a normal scalar
        # `property float color`, not as RGBA channels
        assert "property uchar red" not in header
        assert "property float color" in header

    def test_ply(self):
        m = g.get_mesh("machinist.XAML")

        assert m.visual.kind == "face"
        assert g.np.ptp(m.visual.face_colors, axis=0).max() > 0

        export = m.export(file_type="ply")
        reconstructed = g.roundtrip(export, file_type="ply")

        assert reconstructed.visual.kind == "face"

        assert g.np.allclose(reconstructed.visual.face_colors, m.visual.face_colors)

        m = g.get_mesh("reference.ply")

        assert m.visual.kind == "vertex"
        assert g.np.ptp(m.visual.vertex_colors, axis=0).max() > 0

        export = m.export(file_type="ply")
        reconstructed = g.roundtrip(export, file_type="ply")
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
        reconstructed = g.roundtrip(export, file_type="ply")

        vertex_attributes = reconstructed.metadata["_ply_raw"]["vertex"]["data"]
        result_1d = vertex_attributes["test_1d_attribute"]
        result_nd = vertex_attributes["test_nd_attribute"]["f1"]

        g.np.testing.assert_almost_equal(result_1d, test_1d_attribute)
        g.np.testing.assert_almost_equal(result_nd, test_nd_attribute)

    def test_face_attributes(self):
        # Test writing face attributes to a ply, by reading
        # them back and asserting the written attributes array matches

        for encoding in ["binary", "ascii"]:
            for dt in [g.np.float32, g.np.float64]:
                m = g.get_mesh("box.STL")
                test_1d_attribute = g.np.copy(m.face_angles[:, 0])
                test_nd_attribute = g.np.copy(m.face_angles)
                m.face_attributes["test_1d_attribute"] = test_1d_attribute.astype(dt)
                m.face_attributes["test_nd_attribute"] = test_nd_attribute.astype(dt)

                export = m.export(
                    file_type="ply", include_attributes=True, encoding=encoding
                )
                reconstructed = g.roundtrip(export, file_type="ply", process=False)

                face_attributes = reconstructed.metadata["_ply_raw"]["face"]["data"]
                result_1d = face_attributes["test_1d_attribute"]
                if encoding == "binary":
                    # only binary format allows this
                    result_nd = face_attributes["test_nd_attribute"]["f1"]
                else:
                    result_nd = face_attributes["test_nd_attribute"]

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
        en = g.roundtrip(mesh.export(file_type="ply", encoding="ascii"), file_type="ply")
        assert en.visual.kind is None

        color = [255, 0, 0, 255]
        mesh.visual.vertex_colors = color

        # try exporting and reloading raw
        eb = g.roundtrip(mesh.export(file_type="ply"), file_type="ply")

        assert g.np.allclose(eb.visual.vertex_colors[0], color)
        assert eb.visual.kind == "vertex"

        ea = g.roundtrip(mesh.export(file_type="ply", encoding="ascii"), file_type="ply")
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
                reconstructed = g.roundtrip(export, file_type="ply")

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

    def test_skip_texturefile(self):
        # not loading the texture should produce a trivial texture
        m_tex = g.get_mesh("fuze.ply")
        m_tex_size = m_tex.visual.material.image.size

        m_notex = g.get_mesh("fuze.ply", skip_materials=True)
        m_notex_size = m_notex.visual.material.image.size

        assert m_tex_size != m_notex_size

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
