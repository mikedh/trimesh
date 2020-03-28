try:
    from . import generic as g
except BaseException:
    import generic as g


class GLTFTest(g.unittest.TestCase):

    def test_duck(self):
        scene = g.get_mesh('Duck.glb', process=False)

        # should have one mesh
        assert len(scene.geometry) == 1

        # get the mesh
        geom = next(iter(scene.geometry.values()))

        # should not be watertight
        assert not geom.is_volume
        # make sure export doesn't crash
        export = scene.export(file_type='glb')
        assert len(export) > 0
        # check a roundtrip
        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(export),
            file_type='glb')
        # make basic assertions
        g.scene_equal(scene, reloaded)

        # if we merge ugly it should now be watertight
        geom.merge_vertices(merge_tex=True)
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
        export = s.export(file_type='glb')
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
        export = original.export(file_type='glb')
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
        export = scene.export(file_type='gltf')
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

        export = scene.export(file_type='glb')
        assert len(export) > 0
        # check a roundtrip
        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(export),
            file_type='glb')
        # make basic assertions
        g.scene_equal(scene, reloaded)

    def test_material_hash(self):

        # load mesh twice independently
        a = g.get_mesh('fuze.obj')
        b = g.get_mesh('fuze.obj')
        # move one of the meshes away from the other
        a.apply_translation([a.scale, 0, 0])

        # materials should not be the same object
        assert id(a.visual.material) != id(b.visual.material)
        # despite being loaded separately material hash should match
        assert hash(a.visual.material) == hash(b.visual.material)

        # create a scene with two meshes
        scene = g.trimesh.Scene([a, b])
        # get the exported GLTF header of a scene with both meshes
        header = g.json.loads(scene.export(
            file_type='gltf')['model.gltf'].decode('utf-8'))
        # header should contain exactly one material
        assert len(header['materials']) == 1
        # both meshes should be contained in the export
        assert len(header['meshes']) == 2

        # get a reloaded version
        reloaded = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(
                scene.export(file_type='glb')),
            file_type='glb')

        # meshes should have survived
        assert len(reloaded.geometry) == 2
        # get meshes back
        ar, br = reloaded.geometry.values()

        # should have been loaded as a PBR material
        assert isinstance(ar.visual.material,
                          g.trimesh.visual.material.PBRMaterial)

        # materials should have the same memory location
        assert id(ar.visual.material) == id(br.visual.material)

        # make sure hash is returning something
        ahash = hash(ar.visual.material)
        # should be returning valid material hashes
        assert isinstance(ahash, int)
        assert ahash != 0

    def test_node_name(self):
        """
        Test to see if node names generally survive
        an export-import cycle.
        """
        # a scene
        s = g.get_mesh('cycloidal.3DXML')
        # export as GLB then re-load
        r = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(
                s.export(file_type='glb')),
            file_type='glb')
        # make sure we have the same geometries before and after
        assert set(s.geometry.keys()) == set(r.geometry.keys())
        # make sure the node names are the same before and after
        assert (set(s.graph.nodes_geometry) ==
                set(r.graph.nodes_geometry))

    def test_schema(self):
        # get a copy of the GLTF schema and do simple checks
        s = g.trimesh.exchange.gltf.get_schema()

        # make sure it has at least the keys we expect
        assert set(s['properties'].keys()).issuperset(
            {'accessors',
             'animations',
             'asset',
             'buffers',
             'bufferViews',
             'cameras',
             'images',
             'materials',
             'meshes',
             'nodes',
             'samplers',
             'scene',
             'scenes',
             'skins',
             'textures',
             'extensions',
             'extras'})

        # lightly check to see that no references exist
        assert '$ref' not in g.json.dumps(s)

    def test_export_custom_attributes(self):
        # Write and read custom vertex attributes to gltf
        sphere = g.trimesh.primitives.Sphere()
        v_count, _ = sphere.vertices.shape

        sphere.vertex_attributes['_CustomFloat32Scalar'] = g.np.random.rand(
            v_count, 1).astype(
            g.np.float32)
        sphere.vertex_attributes['_CustomUIntScalar'] = g.np.random.randint(
            0, 1000, size=(v_count, 1)
        ).astype(g.np.uintc)
        sphere.vertex_attributes['_CustomFloat32Vec3'] = g.np.random.rand(
            v_count, 3).astype(g.np.float32)
        sphere.vertex_attributes['_CustomFloat32Mat4'] = g.np.random.rand(
            v_count, 4, 4).astype(g.np.float32)

        # export as GLB then re-load
        r = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(
                sphere.export(file_type='glb')),
            file_type='glb')

        for _, val in r.geometry.items():
            assert set(
                val.vertex_attributes.keys()) == set(
                sphere.vertex_attributes.keys())
            for key in val.vertex_attributes:
                is_same = g.np.array_equal(
                    val.vertex_attributes[key],
                    sphere.vertex_attributes[key])
                assert is_same is True

    def test_extras(self):
        # if GLTF extras are defined, make sure they survive a round trip
        s = g.get_mesh('cycloidal.3DXML')

        # some dummy data
        dummy = {'who': 'likes cheese', 'potatoes': 25}

        # export as GLB with extras passed to the exporter then re-load
        r = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(
                s.export(file_type='glb', extras=dummy)),
            file_type='glb')

        # shouldn't be in original metadata
        assert 'extras' not in s.metadata
        # make sure extras survived a round trip
        assert all(r.metadata['extras'][k] == v
                   for k, v in dummy.items())

        # now assign the extras to the metadata
        s.metadata['extras'] = dummy
        # export as GLB then re-load
        r = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(
                s.export(file_type='glb')),
            file_type='glb')
        # make sure extras survived a round trip
        assert all(r.metadata['extras'][k] == v
                   for k, v in dummy.items())


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
