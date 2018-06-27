try:
    from . import generic as g
except BaseException:
    import generic as g


class DecompositionTest(g.unittest.TestCase):

    def test_convex_decomposition(self):
        mesh = g.get_mesh('quadknot.obj')

        engines = [('vhacd', g.trimesh.interfaces.vhacd.exists)]

        for engine, exists in engines:
            if not exists:
                g.log.warning(
                    'skipping convex decomposition engine %s', engine)
                continue

            g.log.info('Testing convex decomposition with engine %s', engine)
            meshes = mesh.convex_decomposition(engine=engine)
            self.assertTrue(len(meshes) > 1)
            for m in meshes:
                self.assertTrue(m.is_watertight)

            g.log.info('convex decomposition succeeded with %s', engine)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
