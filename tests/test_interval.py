try:
    from . import generic as g
except BaseException:
    import generic as g


class IntervalTest(g.unittest.TestCase):

    def test_intersection(self):

        pairs = g.np.array([[[0, 1], [1, 2]],
                            [[1, 0], [1, 2]],
                            [[0, 0], [0, 0]],
                            [[10, 20], [9, 21]],
                            [[5, 15], [7, 10]],
                            [[5, 10], [10, 9]],
                            [[0, 1], [0.9, 10]]])
        tru_hit = [False,
                   False,
                   False,
                   True,
                   True,
                   True,
                   True]
        tru_int = g.np.array([[0.0, 0.0],
                              [0.0, 0.0],
                              [0.0, 0.0],
                              [10, 20],
                              [7, 10],
                              [9, 10],
                              [0.9, 1.0]])

        func = g.trimesh.interval.intersection

        # check the single- interval results
        for ab, h, i in zip(pairs, tru_hit, tru_int):
            r_h, r_i = func(*ab)
            assert g.np.allclose(r_i, i)
            assert r_h == h

        # check the vectorized multiple interval results
        r_h, r_i = func(pairs[:, 0, :], pairs[:, 1, :])
        assert g.np.allclose(r_h, tru_hit)
        assert g.np.allclose(r_i, tru_int)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
