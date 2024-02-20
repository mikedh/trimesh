try:
    from . import generic as g
except BaseException:
    import generic as g


class NSphereTest(g.unittest.TestCase):
    def test_minball(self):
        # how close do we need to be
        tol_fit = 1e-2

        # get some assorted mesh geometries to test performance
        # and a perfect sphere mesh to test the degenerate case
        for m in g.np.append(list(g.get_meshes(5)), g.trimesh.primitives.Sphere()):
            s = m.bounding_sphere
            R_check = ((m.vertices - s.primitive.center) ** 2).sum(axis=1).max() ** 0.5

            assert len(s.primitive.center) == 3
            assert s.primitive.radius > 0.0
            assert abs(s.primitive.radius - R_check) < tol_fit
            assert s.volume > (m.volume - tol_fit)

        # check minimum n-sphere for points in 2, 3, 4 dimensions
        for d in [2, 3, 4]:
            for _i in range(5):
                points = g.random((100, d))
                C, R = g.trimesh.nsphere.minimum_nsphere(points)
                R_check = ((points - C) ** 2).sum(axis=1).max() ** 0.5
                assert len(C) == d
                assert R > 0.0
                assert abs(R - R_check) < g.tol.merge

    def test_isnsphere(self):
        # make sure created spheres are uv sphere
        m = g.trimesh.creation.uv_sphere()
        # move the mesh around for funsies
        m.apply_translation(g.random(3))
        m.apply_transform(g.trimesh.transformations.random_rotation_matrix())
        # all vertices should be on nsphere
        assert g.trimesh.nsphere.is_nsphere(m.vertices)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
