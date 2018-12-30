try:
    from . import generic as g
except BaseException:
    import generic as g


class GLTFTest(g.unittest.TestCase):

    def test_duck(self):
        scene = g.get_mesh('Duck.glb')

        assert len(scene.geometry) == 1
        
        for name, model in scene.geometry.items():
            assert model.is_volume

    def test_cesium(self):
        """
        A GLTF with a multi- primitive mesh
        """
        s = g.get_mesh('CesiumMilkTruck.glb')
        # should be one Trimesh object per GLTF "primitive"
        assert len(s.geometry) == 4
        # every geometry displayed once, except wheels twice
        assert len(s.graph.nodes_geometry) == 5

        
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
        reloaded = g.trimesh.exchange.load.load_kwargs(kwargs)

        # make assertions on original and reloaded
        for scene in [original, reloaded]:
            # units should be stored as an extra
            assert scene.units == 'mm'

            # make sure we have two unique geometries
            assert len(scene.geometry) == 2
            # that should have seven instances
            assert len(scene.graph.nodes_geometry) == 7

            # all meshes should be well constructed
            assert all(m.is_volume for m in scene.geometry.values())

            # check unit conversions for fun
            extents = scene.extents.copy()
            as_in = scene.convert_units('in')
            # should all be exactly mm -> in conversion factor
            assert g.np.allclose(extents / as_in.extents, 25.4, atol=.001)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
