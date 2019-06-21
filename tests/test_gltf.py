try:
    from . import generic as g
except BaseException:
    import generic as g


class GLTFTest(g.unittest.TestCase):

    def test_duck(self):
        scene = g.get_mesh('Duck.glb')

        # should have one mesh
        assert len(scene.geometry) == 1

        # get the mesh
        geom = next(iter(scene.geometry.values()))
        # should not be watertight
        assert not geom.is_volume
        # make sure export doesn't crash
        export = scene.export('glb')
        assert len(export) > 0
        # check a roundtrip
        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(export),
            file_type='glb')
        # make basic assertions
        g.scene_equal(scene, reloaded)

        # if we merge ugly it should now be watertight
        geom.merge_vertices(textured=False)
        assert geom.is_volume

    def test_tex_export(self):
        # load textured PLY
        mesh = g.get_mesh('fuze.ply')
        assert hasattr(mesh.visual, 'uv')

        # make sure export as GLB doesn't crash on scenes
        export = mesh.scene().export(file_type='glb')
        assert len(export) > 0
        # make sure it works on meshes
        export = mesh.export(file_type='glb')
        assert len(export) > 0

    def test_cesium(self):
        """
        A GLTF with a multi- primitive mesh
        """
        s = g.get_mesh('CesiumMilkTruck.glb')
        # should be one Trimesh object per GLTF "primitive"
        assert len(s.geometry) == 4
        # every geometry displayed once, except wheels twice
        assert len(s.graph.nodes_geometry) == 5

        # make sure export doesn't crash
        export = s.export('glb')
        assert len(export) > 0

        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(export),
            file_type='glb')
        # make basic assertions
        g.scene_equal(s, reloaded)

    def test_units(self):
        """
        Trimesh will store units as a GLTF extra if they
        are defined so check that.
        """
        original = g.get_mesh('pins.glb')

        # export it as a a GLB file
        export = original.export('glb')
        kwargs = g.trimesh.exchange.gltf.load_glb(
            g.trimesh.util.wrap_as_stream(export))
        # roundtrip it
        reloaded = g.trimesh.exchange.load.load_kwargs(kwargs)
        # make basic assertions
        g.scene_equal(original, reloaded)

        # make assertions on original and reloaded
        for scene in [original, reloaded]:
            # units should be stored as an extra
            assert scene.units == 'mm'

            # make sure we have two unique geometries
            assert len(scene.geometry) == 2
            # that should have seven instances
            assert len(scene.graph.nodes_geometry) == 7

            # all meshes should be well constructed
            assert all(m.is_volume for m in
                       scene.geometry.values())

            # check unit conversions for fun
            extents = scene.extents.copy()
            as_in = scene.convert_units('in')
            # should all be exactly mm -> in conversion factor
            assert g.np.allclose(
                extents / as_in.extents, 25.4, atol=.001)

        m = g.get_mesh('testplate.glb')
        assert m.units == 'meters'

    def test_gltf(self):
        # split a multibody mesh into a scene
        scene = g.trimesh.scene.split_scene(
            g.get_mesh('cycloidal.ply'))
        # should be 117 geometries
        assert len(scene.geometry) >= 117

        # a dict with {file name: str}
        export = scene.export('gltf')
        # load from just resolver
        r = g.trimesh.load(file_obj=None,
                           file_type='gltf',
                           resolver=export)
        # will assert round trip is roughly equal
        g.scene_equal(r, scene)

        # try loading from a ZIP archive
        zipped = g.trimesh.util.compress(export)
        r = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(zipped),
            file_type='zip')

        # try loading from a file name
        # will require a file path resolver
        with g.TemporaryDirectory() as d:
            for file_name, data in export.items():
                with open(g.os.path.join(d, file_name), 'wb') as f:
                    f.write(data)
            # load from file path of header GLTF
            rd = g.trimesh.load(
                g.os.path.join(d, 'model.gltf'))
            # will assert round trip is roughly equal
            g.scene_equal(rd, scene)

    def test_gltf_pole(self):
        scene = g.get_mesh('simple_pole.glb')

        # should have multiple primitives
        assert len(scene.geometry) == 11

        export = scene.export('glb')
        assert len(export) > 0
        # check a roundtrip
        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(export),
            file_type='glb')
        # make basic assertions
        g.scene_equal(scene, reloaded)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
