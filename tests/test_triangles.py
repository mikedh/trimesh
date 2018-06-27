try:
    from . import generic as g
except BaseException:
    import generic as g


class TrianglesTest(g.unittest.TestCase):

    def test_barycentric(self):
        for m in g.get_meshes(4):
            # a simple test which gets the barycentric coordinate at each of the three
            # vertices, checks to make sure the barycentric is [1,0,0] for the vertex
            # and then converts back to cartesian and makes sure the original points
            #  are the same as the conversion and back
            for method in ['cross', 'cramer']:
                for i in range(3):
                    barycentric = g.trimesh.triangles.points_to_barycentric(
                        m.triangles, m.triangles[:, i], method=method)
                    self.assertTrue(
                        (g.np.abs(barycentric - g.np.roll([1.0, 0, 0], i)) < 1e-8).all())

                    points = g.trimesh.triangles.barycentric_to_points(
                        m.triangles, barycentric)
                    self.assertTrue(
                        (g.np.abs(points - m.triangles[:, i]) < 1e-8).all())

    def test_closest(self):
        closest = g.trimesh.triangles.closest_point(
            triangles=g.data['triangles']['triangles'],
            points=g.data['triangles']['points'])

        comparison = (closest - g.data['triangles']['closest']).all()

        self.assertTrue((comparison < 1e-8).all())
        g.log.info('finished closest check on %d triangles', len(closest))

    def test_degenerate(self):
        tri = [[[0, 0, 0],
                [1, 0, 0],
                [-.5, 0, 0]],
               [[0, 0, 0],
                [0, 0, 0],
                [10, 10, 0]],
               [[0, 0, 0],
                [0, 0, 2],
                [0, 0, 2.2]],
               [[0, 0, 0],
                [1, 0, 0],
                [0, 1, 0]]]

        tri_gt = [False,
                  False,
                  False,
                  True]

        r = g.trimesh.triangles.nondegenerate(tri)
        self.assertTrue(len(r) == len(tri))
        self.assertTrue((r == tri_gt).all())


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
