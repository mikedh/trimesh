try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import triangle  # NOQA
    has_triangle = True
except ImportError:
    g.log.warning('No triangle! Not testing extrude primitives!')
    has_triangle = False


class ExtrudeTest(g.unittest.TestCase):

    def test_extrusion(self):
        if not has_triangle:
            return

        transform = g.trimesh.transformations.random_rotation_matrix()
        polygon = g.Point([0, 0]).buffer(.5)
        e = g.trimesh.primitives.Extrusion(
            polygon=polygon,
            transform=transform)

        # will create an inflated version of the extrusion
        b = e.buffer(.1)
        assert b.to_mesh().volume > e.to_mesh().volume
        assert b.contains(e.vertices).all()

        # try making it smaller
        b = e.buffer(-.1)
        assert b.to_mesh().volume < e.to_mesh().volume
        assert e.contains(b.vertices).all()

        # try with negative height
        e = g.trimesh.primitives.Extrusion(
            polygon=polygon,
            height=-1.0,
            transform=transform)
        assert e.to_mesh().volume > 0.0

        # will create an inflated version of the extrusion
        b = e.buffer(.1)
        assert b.to_mesh().volume > e.to_mesh().volume
        assert b.contains(e.vertices).all()

        # try making it smaller
        b = e.buffer(-.1)
        assert b.to_mesh().volume < e.to_mesh().volume
        assert e.contains(b.vertices).all()

        # try with negative height and transform
        transform = [[1., 0., 0., -0.],
                     [0., 1., 0., 0.],
                     [-0., -0., -1., -0.],
                     [0., 0., 0., 1.]]
        e = g.trimesh.primitives.Extrusion(
            polygon=polygon,
            height=-1.0,
            transform=transform)
        assert e.to_mesh().volume > 0.0


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
