try:
    from . import generic as g
except BaseException:
    import generic as g


class IntervalTest(g.unittest.TestCase):
    def test_intersection(self):
        pairs = g.np.array(
            [
                [[0, 1], [1, 2]],
                [[1, 0], [1, 2]],
                [[0, 0], [0, 0]],
                [[10, 20], [9, 21]],
                [[5, 15], [7, 10]],
                [[5, 10], [10, 9]],
                [[0, 1], [0.9, 10]],
                [[1000, 1001], [2000, 2001]],
            ],
            dtype=g.np.float64,
        )

        # true intersection ranges
        truth = g.np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [10, 20],
                [7, 10],
                [9, 10],
                [0.9, 1.0],
                [0, 0],
            ],
            dtype=g.np.float64,
        )

        intersection = g.trimesh.interval.intersection
        union = g.trimesh.interval.union

        # check the single- interval results
        for ab, tru in zip(pairs, truth):
            result = intersection(*ab)
            assert g.np.allclose(result, tru)

        # check the vectorized multiple interval results
        inter = intersection(pairs[:, 0, :], pairs[:, 1, :])

        assert g.np.allclose(truth, inter)

        # now just run a union on these for the fun of it
        u = union(pairs.reshape((-1, 2)))
        assert g.np.allclose(u, [[0.0, 21.0], [1000.0, 1001.0], [2000.0, 2001.0]])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
