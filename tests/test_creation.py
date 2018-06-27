try:
    from . import generic as g
except BaseException:
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

    def test_path_extrude(self):
        try:
            import meshpy
        except ImportError:
            g.log.error("no meshpy: skipping test")
            return

        # Create base polygon
        vec = g.np.array([0, 1]) * 0.2
        n_comps = 100
        angle = g.np.pi * 2.0 / n_comps
        rotmat = g.np.array([
            [g.np.cos(angle), -g.np.sin(angle)],
            [g.np.sin(angle), g.np.cos(angle)]])
        perim = []
        for i in range(n_comps):
            perim.append(vec)
            vec = g.np.dot(rotmat, vec)
        poly = g.Polygon(perim)

        # Create 3D path
        angles = g.np.linspace(0, 8 * g.np.pi, 1000)
        x = angles / 10.0
        y = g.np.cos(angles)
        z = g.np.sin(angles)
        path = g.np.c_[x, y, z]

        # Extrude
        mesh = g.trimesh.creation.sweep_polygon(poly, path)
        self.assertTrue(mesh.is_volume)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
