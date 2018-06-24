try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.transforms import EnforcedForest


def random_chr():
    return chr(ord('a') + int(round(g.np.random.random() * 25)))


class SceneTests(g.unittest.TestCase):

    def test_scene(self):
        for mesh in g.get_mesh('cycloidal.ply',
                               'sphere.ply'):
            if mesh.units is None:
                mesh.units = 'in'

            scene_split = g.trimesh.scene.split_scene(mesh)
            scene_split.convert_units('in')
            scene_base = g.trimesh.Scene(mesh)

            for s in [scene_split, scene_base]:
                self.assertTrue(len(s.geometry) > 0)

                flattened = s.graph.to_flattened()
                g.json.dumps(flattened)
                edgelist = s.graph.to_edgelist()
                g.json.dumps(edgelist)

                assert s.bounds.shape == (2, 3)
                assert s.centroid.shape == (3,)
                assert s.extents.shape == (3,)
                assert isinstance(s.scale, float)
                assert g.trimesh.util.is_shape(s.triangles, (-1, 3, 3))
                assert len(s.triangles) == len(s.triangles_node)

                assert s.md5() is not None

                assert len(s.duplicate_nodes) > 0

                r = s.dump()

                gltf = s.export(file_type='gltf')
                assert isinstance(gltf, dict)
                assert len(gltf) > 0
                assert len(gltf['model.gltf']) > 0

                glb = s.export(file_type='glb')
                assert len(glb) > 0
                assert isinstance(glb, bytes)

                for export_format in ['dict', 'dict64']:
                    # try exporting the scene as a dict
                    # then make sure json can serialize it
                    e = g.json.dumps(s.export(export_format))

                    # reconstitute the dict into a scene
                    r = g.trimesh.load(g.json.loads(e))

                    # make sure the extents are similar before and after
                    assert g.np.allclose(g.np.product(s.extents),
                                         g.np.product(r.extents))

                s.rezero()
                assert (g.np.abs(s.centroid) < 1e-3).all()

                # make sure explode doesn't crash
                s.explode()

    def test_scaling(self):
        '''
        Test the scaling of scenes including unit conversion.
        '''
        scene = g.get_mesh('cycloidal.3DXML')

        md5 = scene.md5()
        extents = scene.bounding_box_oriented.primitive.extents.copy()

        factor = 10.0
        scaled = scene.scaled(factor)

        # the oriented bounding box should scale exactly with the scaling
        # factor
        assert g.np.allclose(
            scaled.bounding_box_oriented.primitive.extents /
            extents,
            factor)

        # we shouldn't have modified the original scene
        assert scene.md5() == md5
        assert scaled.md5() != md5

        # 3DXML comes in as mm
        assert all(m.units == 'mm' for m in scene.geometry.values())

        converted = scene.convert_units('in')

        assert g.np.allclose(
            converted.bounding_box_oriented.primitive.extents /
            extents,
            1.0 /
            25.4)

        assert all(m.units == 'in' for m in converted.geometry.values())

        # we shouldn't have modified the original scene
        assert scene.md5() == md5
        assert converted.md5() != md5

        populate = scene.bounding_box

    def test_3DXML(self):
        s = g.get_mesh('rod.3DXML')

        assert len(s.geometry) == 3
        assert len(s.graph.nodes_geometry) == 29

    def test_empty(self):
        m = g.get_mesh('bunny.ply')

        # not watertight so will result in empty scene
        s = g.trimesh.scene.split_scene(m)
        assert len(s.geometry) == 0

        s = s.convert_units('inches')
        n = s.duplicate_nodes
        assert len(n) == 0


class GraphTests(g.unittest.TestCase):

    def test_forest(self):
        g = EnforcedForest(assert_forest=True)
        for i in range(5000):
            g.add_edge(random_chr(), random_chr())


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
