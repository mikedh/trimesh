import generic as g


class CreationTest(g.unittest.TestCase):

    def test_soup(self):
        count = 100
        mesh = g.trimesh.creation.random_soup(face_count=count)

        self.assertTrue(len(mesh.faces) == count)
        self.assertTrue(len(mesh.face_adjacency) == 0)
        self.assertTrue(len(mesh.split(only_watertight=True)) == 0)
        self.assertTrue(len(mesh.split(only_watertight=False)) == count)

    def test_uv(self):
        sphere = g.trimesh.creation.uv_sphere()
        self.assertTrue(sphere.is_watertight)
        self.assertTrue(sphere.is_winding_consistent)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
