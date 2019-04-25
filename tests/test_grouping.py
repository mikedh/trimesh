try:
    from . import generic as g
except BaseException:
    import generic as g


class GroupTests(g.unittest.TestCase):

    def test_unique_rows(self):
        count = 10000
        subset = int(count / 10)

        # check unique_rows on float data
        data = g.np.arange(count * 3).reshape((-1, 3)).astype(g.np.float)
        data[:subset] = data[0]
        unique, inverse = g.trimesh.grouping.unique_rows(data)
        assert (inverse[:subset] == 0).all()
        assert len(unique) == count - subset + 1

        # check the bitbanging path of hashable rows on small integers
        data = data[:, :2].astype(int)
        unique, inverse = g.trimesh.grouping.unique_rows(data)
        assert (inverse[:subset] == 0).all()
        assert len(unique) == count - subset + 1

    def test_blocks(self):
        blocks = g.trimesh.grouping.blocks

        count = 100
        subset = int(count / 10)
        a = g.np.zeros(count, dtype=g.np.int)

        result = blocks(a, min_len=0, only_nonzero=False)
        assert len(result) == 1
        assert len(result[0]) == count

        result = blocks(a, min_len=0, only_nonzero=True)
        assert len(result) == 0

        result = blocks(a, min_len=count + 1, only_nonzero=False)
        assert len(result) == 0

        result = blocks(a, max_len=count - 1, only_nonzero=False)
        assert len(result) == 0

        result = blocks(a, max_len=count + 1, only_nonzero=False)
        assert len(result) == 1
        assert len(result[0]) == count

        a[:subset] = True
        result = blocks(a, only_nonzero=False)
        assert len(result) == 2
        assert set(range(subset)) == set(result[0])
        assert set(range(subset, count)) == set(result[1])
        assert sum(len(i) for i in result) == count

        result = blocks(a, only_nonzero=True)
        assert len(result) == 1
        assert set(range(subset)) == set(result[0])
        assert all(a[i].all() for i in result)

        a[0] = False
        result = blocks(a, min_len=1, only_nonzero=True)
        assert len(result) == 1
        assert set(range(1, subset)) == set(result[0])
        assert all(a[i].all() for i in result)

        result = blocks(a, min_len=1, only_nonzero=False)
        assert len(result) == 3
        assert sum(len(i) for i in result) == count

        a[2] = False
        result = blocks(a, min_len=1, only_nonzero=True)
        assert len(result) == 2
        assert set(result[0]) == set([1])
        assert all(a[i].all() for i in result)

    def test_runs(self):
        a = g.np.array([-1, -1, -1, 0, 0, 1, 1, 2,
                        0, 3, 3, 4, 4, 5, 5, 6,
                        6, 7, 7, 8, 8, 9, 9, 9],
                       dtype=g.np.int)
        r = g.trimesh.grouping.merge_runs(a)
        u = g.trimesh.grouping.unique_ordered(a)

        self.assertTrue((g.np.diff(r) != 0).all())
        self.assertTrue((g.np.diff(u) != 0).all())

        self.assertTrue(r.size == 12)
        self.assertTrue(u.size == 11)

    def test_cluster(self):
        a = (g.np.random.random((10000, 3)) * 5).astype(int)

        r = g.trimesh.grouping.clusters(a, .01)  # NOQA

        r = g.trimesh.grouping.group_distance(a, .01)  # NOQA

    def test_unique_float(self):

        a = g.np.arange(100) / 2.0
        t = g.np.tile(a, 2).flatten()

        unique = g.trimesh.grouping.unique_float(t)
        assert g.np.allclose(unique, a)

        unique, index, inverse = g.trimesh.grouping.unique_float(t,
                                                                 return_index=True,
                                                                 return_inverse=True)
        assert g.np.allclose(unique[inverse], t)
        assert g.np.allclose(unique, t[index])

    def test_group_rows(self):
        a = g.np.arange(100) / 2.0
        b = g.np.tile(a, 3).reshape((-1, 3))
        c = g.np.vstack((b, b))

        gr = g.trimesh.grouping.group_rows(c)
        assert gr.shape == (100, 2)
        assert g.np.allclose(c[gr].ptp(axis=1), 0.0)

        gr = g.trimesh.grouping.group_rows(c, require_count=2)
        assert gr.shape == (100, 2)
        assert g.np.allclose(c[gr].ptp(axis=1), 0.0)

        c = g.np.vstack((c, [1, 2, 3]))
        gr = g.trimesh.grouping.group_rows(c, require_count=2)
        grd = g.trimesh.grouping.group_rows(c)
        # should discard the single element
        assert gr.shape == (100, 2)
        # should get the single element correctly
        assert len(grd) == 101
        assert sum(1 for i in grd if len(i) == 2) == 100
        assert g.np.allclose(c[gr].ptp(axis=1), 0.0)

    def test_group_vector(self):
        x = g.np.linspace(-100, 100, 100)

        vec = g.np.column_stack((x,
                                 g.np.ones(len(x)),
                                 g.np.zeros(len(x))))
        vec = g.trimesh.unitize(vec)

        uv, ui = g.trimesh.grouping.group_vectors(vec)
        assert g.np.allclose(uv, vec)
        assert len(vec) == len(ui)
        assert g.np.allclose(uv[ui.flatten()], vec)

        vec = g.np.vstack((vec, -vec))
        uv, ui = g.trimesh.grouping.group_vectors(vec)
        assert g.np.allclose(uv, vec)
        assert len(ui) == len(vec)

        uv, ui = g.trimesh.grouping.group_vectors(vec,
                                                  include_negative=True)
        # since we included negative vectors, there should
        # be half the number of unique vectors and 2 indexes per vector
        assert ui.shape == (100, 2)
        assert uv.shape == (100, 3)
        assert g.np.allclose(uv, vec[:100])

    def test_boolean_rows(self):
        a = g.np.arange(10).reshape((-1, 2))
        b = g.np.arange(10).reshape((-1, 2)) + 8
        # make one a different dtype
        b = b.astype(g.np.int32)

        # should have one overlapping row
        intersection = g.trimesh.grouping.boolean_rows(
            a, b, g.np.intersect1d)
        assert g.np.allclose(intersection.ravel(), [8, 9])

        diff = g.trimesh.grouping.boolean_rows(
            a, b, g.np.setdiff1d)
        assert g.np.allclose(g.np.unique(diff),
                             g.np.arange(8))


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
