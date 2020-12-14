try:
    from . import generic as g
except BaseException:
    import generic as g


class VHACDTest(g.unittest.TestCase):

    def test_vhacd(self):

        # exit if no VHACD
        if not g.trimesh.interfaces.vhacd.exists:
            g.log.warning(
                'not testing convex decomposition (no vhacd)!')
            return

        g.log.info('testing convex decomposition using vhacd')
        # get a bunny
        mesh = g.get_mesh('bunny.ply')
        # run a convex decomposition using vhacd
        decomposed = mesh.convex_decomposition(
            maxhulls=10, debug=True)

        if len(decomposed) != 10:
            # it should return the correct number of meshes
            raise ValueError('{} != 10'.format(len(decomposed)))

        # make sure everything is convex
        # also this will fail if the type is returned incorrectly
        assert all(i.is_convex for i in decomposed)

        # make sure every result is actually a volume
        # ie watertight, consistent winding, positive nonzero volume
        assert all(i.is_volume for i in decomposed)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
