try:
    from . import generic as g
except BaseException:
    import generic as g


class UnwrapTest(g.unittest.TestCase):
    def setUp(self):
        self.a = g.get_mesh('bunny.ply', force="mesh")
        self.truth = {}
        for k in ["bounds",
                  "extents",
                  "scale",
                  "centroid",
                  "center_mass",
                  "density",
                  "volume",
                  "mass",
                  "moment_inertia",
                  "principal_inertia_components",
                  "principal_inertia_vectors",
                  "principal_inertia_transform",
                  "area"]:
            self.truth[k] = getattr(self.a, k)

        self.engine = 'blender'

    def test_unwrap(self):
        a = self.a

        if not g.trimesh.interfaces.blender.exists:
            g.log.warning('skipping unwrap engine %s', self.engine)
            return

        g.log.info('Testing unwrap ops with engine %s', self.engine)
        u = a.unwrap()

        for k, truth in self.truth.items():
            g.np.testing.assert_allclose(getattr(u, k), truth,
                                         rtol=5e-1, atol=1e-6, err_msg=k)

        g.log.info('unwrap succeeded with %s', self.engine)

    def test_image(self):
        a = self.a

        if not g.trimesh.interfaces.blender.exists:
            g.log.warning('skipping unwrap engine %s', self.engine)
            return

        g.log.info('Testing unwrap ops with engine %s', self.engine)

        u = a.unwrap()
        self.assertEqual(u.visual.material.image is None,
                         not hasattr(a.visual, "material") or
                         not hasattr(a.visual.material, "image") or
                         a.visual.material.image is None)

        checkerboard = g.np.kron([[1, 0] * 4, [0, 1] * 4] * 4, g.np.ones((10, 10)))
        u = a.unwrap(image=(checkerboard * 255).astype(g.np.uint8))
        self.assertIsNotNone(u.visual.material.image)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
