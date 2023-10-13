try:
    from . import generic as g
except BaseException:
    import generic as g

import io


class ExportTest(g.unittest.TestCase):
    def test_export(self):
        from trimesh.exceptions import ExceptionWrapper

        export_types = {
            k
            for k, v in g.trimesh.exchange.export._mesh_exporters.items()
            if not isinstance(v, ExceptionWrapper)
        }

        meshes = list(g.get_meshes(8))
        # make sure we've got something with texture
        meshes.append(g.get_mesh("fuze.obj"))

        for mesh in meshes:
            # disregard texture
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            for file_type in export_types:
                # skip pointcloud format
                if file_type in ["xyz", "gltf"]:
                    # a pointcloud format
                    continue
                # run the export
                export = mesh.export(file_type=file_type)
                # if nothing returned log the message
                if export is None or len(export) == 0:
                    raise ValueError(
                        "No data exported %s to %s", mesh.metadata["file_name"], file_type
                    )

                if file_type in [
                    "dae",  # collada, no native importers
                    "collada",  # collada, no native importers
                    "msgpack",  # kind of flaky, but usually works
                    "drc",
                ]:  # DRC is not a lossless format
                    g.log.warning("no native loaders implemented for collada!")
                    continue

                g.log.info("Export/import testing on %s", mesh.metadata["file_name"])

                # if export is string or bytes wrap as pseudo file object
                if isinstance(export, str) or isinstance(export, bytes):
                    file_obj = g.io_wrap(export)
                else:
                    file_obj = export

                loaded = g.trimesh.load(
                    file_obj=file_obj,
                    file_type=file_type,
                    process=True,
                    merge_norm=True,
                    merge_tex=True,
                )

                # if we exported as GLTF/dae it will come back as a Scene
                if isinstance(loaded, g.trimesh.Scene) and isinstance(
                    mesh, g.trimesh.Trimesh
                ):
                    assert len(loaded.geometry) == 1
                    loaded = next(iter(loaded.geometry.values()))

                if (
                    not g.trimesh.util.is_shape(loaded._data["faces"], (-1, 3))
                    or not g.trimesh.util.is_shape(loaded._data["vertices"], (-1, 3))
                    or loaded.faces.shape != mesh.faces.shape
                ):
                    g.log.error(
                        "Export -> import for %s on %s wrong shape!",
                        file_type,
                        mesh.metadata["file_name"],
                    )

                if loaded.vertices is None:
                    g.log.error(
                        "Export -> import for %s on %s gave None for vertices!",
                        file_type,
                        mesh.metadata["file_name"],
                    )

                if loaded.faces.shape != mesh.faces.shape:
                    raise ValueError(
                        "export cycle {} on {} gave faces {}->{}!".format(
                            file_type,
                            mesh.metadata["file_name"],
                            str(mesh.faces.shape),
                            str(loaded.faces.shape),
                        )
                    )

                if loaded.vertices.shape != mesh.vertices.shape:
                    raise ValueError(
                        "export cycle {} on {} gave vertices {}->{}!".format(
                            file_type,
                            mesh.metadata["file_name"],
                            mesh.vertices.shape,
                            loaded.vertices.shape,
                        )
                    )

                # try exporting/importing certain file types by name
                if file_type in ["obj", "stl", "ply", "off"]:
                    temp = g.tempfile.NamedTemporaryFile(
                        suffix="." + file_type, delete=False
                    )
                    # windows throws permissions errors if you keep it open
                    temp.close()

                    mesh.export(temp.name)
                    load = g.trimesh.load(temp.name)
                    # manual cleanup
                    g.os.remove(temp.name)

                    assert mesh.faces.shape == load.faces.shape
                    assert mesh.vertices.shape == load.vertices.shape

            # if we're not on linux don't run meshlab tests
            if not g.is_linux:
                continue
            # formats exportable by trimesh and importable by meshlab
            # make sure things we export can be loaded by meshlab
            both = set(g.meshlab_formats).intersection(set(export_types))

            # additional options to pass to exporters to try to ferret
            # out combinations which lead to invalid output
            kwargs = {
                "ply": [
                    {"vertex_normal": True, "encoding": "ascii"},
                    {"vertex_normal": True, "encoding": "binary"},
                    {"vertex_normal": False, "encoding": "ascii"},
                    {"vertex_normal": False, "encoding": "binary"},
                ],
                "stl": [{"file_type": "stl"}, {"file_type": "stl_ascii"}],
            }

            # make sure input mesh has garbage removed
            mesh._validate = True
            # since we're going to be looking for exact export
            # counts remove anything small/degenerate again
            mesh.process()

            # run through file types supported by both meshlab and trimesh
            for file_type in both:
                # pull different exporter options for the format
                if file_type in kwargs:
                    options = kwargs[file_type]
                else:
                    options = [{}]

                # try each combination of options
                for option in options:
                    temp = g.tempfile.NamedTemporaryFile(
                        suffix="." + file_type, delete=False
                    )
                    temp_off = g.tempfile.NamedTemporaryFile(suffix=".off", delete=False)
                    # windows throws permissions errors if you keep it open
                    temp.close()
                    temp_off.close()
                    # write over the tempfile
                    option["file_obj"] = temp.name
                    mesh.export(**option)

                    # -_-
                    ms = g.pymeshlab.MeshSet()
                    ms.load_new_mesh(temp.name)
                    ms.save_current_mesh(temp_off.name)

                    # load meshlabs export back into trimesh
                    r = g.trimesh.load(temp_off.name)

                    # we should have the same number of vertices and faces
                    assert len(r.vertices) == len(mesh.vertices)
                    assert len(r.faces) == len(mesh.faces)

                    # manual cleanup
                    g.os.remove(temp.name)
                    g.os.remove(temp_off.name)

    def test_obj(self):
        m = g.get_mesh("textured_tetrahedron.obj", process=False)
        export = m.export(file_type="obj")
        reconstructed = g.wrapload(export, file_type="obj", process=False)
        # test that we get at least the same number of normals and texcoords out;
        # the loader may reorder vertices, so we shouldn't check direct
        # equality
        assert m.vertex_normals.shape == reconstructed.vertex_normals.shape

    def test_obj_order(self):
        """
        Make sure simple round trips through Wavefront don't
        reorder vertices.
        """
        # get a writeable temp file location
        temp = g.tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        temp.close()

        # simple solid
        x = g.trimesh.creation.icosahedron()
        x.export(temp.name)
        y = g.trimesh.load_mesh(temp.name, process=False)

        # vertices should be same shape and order
        assert g.np.allclose(x.vertices, y.vertices)
        # faces should be same
        assert g.np.allclose(x.faces, y.faces)

    def test_dict(self):
        mesh = g.get_mesh("machinist.XAML")
        assert mesh.visual.kind == "face"
        mesh.visual.vertex_colors = mesh.visual.vertex_colors
        assert mesh.visual.kind == "vertex"

        as_dict = mesh.to_dict()
        back = g.trimesh.Trimesh(**as_dict)  # NOQA

    def test_scene(self):
        # get a multi- mesh scene with a transform tree
        source = g.get_mesh("cycloidal.3DXML")
        # add a transform to zero scene before exporting
        source.rezero()
        # export the file as a binary GLTF file, GLB
        export = source.export(file_type="glb")

        # re- load the file as a trimesh.Scene object again
        loaded = g.wrapload(export, file_type="glb")

        # the scene should be identical after export-> import cycle
        assert g.np.allclose(loaded.extents / source.extents, 1.0)

    def test_gltf_path(self):
        """
        Check to make sure GLTF exports of Path2D and Path3D
        objects don't immediately crash.
        """
        path2D = g.get_mesh("2D/wrench.dxf")
        path3D = path2D.to_3D()

        a = g.trimesh.Scene(path2D).export(file_type="glb")
        b = g.trimesh.Scene(path3D).export(file_type="glb")

        assert len(a) > 0
        assert len(b) > 0

    def test_parse_file_args(self):
        """
        Test the magical trimesh.exchange.load.parse_file_args
        """
        # it's wordy
        f = g.trimesh.exchange.load.parse_file_args

        RET_COUNT = 5

        # a path that doesn't exist
        nonexists = f"/banana{g.random()}"
        assert not g.os.path.exists(nonexists)

        # loadable OBJ model
        exists = g.os.path.join(g.dir_models, "tube.obj")
        assert g.os.path.exists(exists)

        # should be able to extract type from passed filename
        args = f(file_obj=exists, file_type=None)
        assert len(args) == RET_COUNT
        assert args[1] == "obj"

        # should be able to extract correct type from longer name
        args = f(file_obj=exists, file_type="YOYOMA.oBj")
        assert len(args) == RET_COUNT
        assert args[1] == "obj"

        # with a nonexistent file and no extension it should raise
        try:
            args = f(file_obj=nonexists, file_type=None)
        except ValueError as E:
            assert "not a file" in str(E)
        else:
            raise ValueError("should have raised exception!")

        # nonexistent file with extension passed should return
        # file name anyway, maybe something else can handle it
        args = f(file_obj=nonexists, file_type=".ObJ")
        assert len(args) == RET_COUNT
        # should have cleaned up case
        assert args[1] == "obj"

        # make sure overriding type works for string filenames
        args = f(file_obj=exists, file_type="STL")
        assert len(args) == RET_COUNT
        # should have used manually passed type over .obj
        assert args[1] == "stl"

    def test_buffered_random(self):
        """Test writing to non-standard file"""
        mesh = list(g.get_meshes(1))[0]
        with io.BufferedRandom(io.BytesIO()) as rw:
            mesh.export(rw, "STL")
            rw.seek(0)
            binary_stl = rw.read()
            self.assertLess(0, len(binary_stl))


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
