import generic as g


class GLTFTest(g.unittest.TestCase):

    def test_duck(self):
        scene = g.get_mesh('Duck.glb')
        for name, model in scene.geometry.items():
            assert model.is_volume

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
