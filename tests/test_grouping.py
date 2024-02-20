try:
    from . import generic as g
except BaseException:
    import generic as g


class GroupTests(g.unittest.TestCase):
    def test_unique_rows(self):
        count = 10000
        subset = int(count / 10)

        # check unique_rows on float data
        data = g.np.arange(count * 3).reshape((-1, 3)).astype(g.np.float64)
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
        """
        Blocks are equivalent values next to each other in
        a 1D array.
        """
        blocks = g.trimesh.grouping.blocks

        count = 100
        subset = int(count / 10)
        a = g.np.zeros(count, dtype=g.np.int64)

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
        assert set(result[0]) == {1}
        assert all(a[i].all() for i in result)

        # make sure wrapping works if all values are True
        arr = g.np.ones(10, dtype=bool)
        result = blocks(arr, min_len=1, wrap=True, only_nonzero=True)
        assert len(result) == 1
        assert set(result[0]) == set(range(10))

        # and all false
        arr = g.np.zeros(10, dtype=bool)
        result = blocks(arr, min_len=1, wrap=True, only_nonzero=True)
        assert len(result) == 0

        arr = g.np.zeros(10, dtype=bool)
        result = blocks(arr, min_len=1, wrap=True, only_nonzero=False)
        assert len(result) == 1
        assert set(result[0]) == set(range(10))

    def test_block_wrap(self):
        """
        Test blocks with wrapping
        """
        # save the mouthful
        blocks = g.trimesh.grouping.blocks

        # case: both ends are in a block
        data = g.np.array([1, 1, 0, 0, 1, 1, 1, 1])
        kwargs = {"data": data, "min_len": 2, "wrap": True, "only_nonzero": True}
        r = blocks(**kwargs)
        # should be one group
        assert len(r) == 1
        # should have every element
        assert g.np.allclose(r[0], [4, 5, 6, 7, 0, 1])
        assert len(r[0]) == data.sum()
        assert g.np.allclose(data[r[0]], 1)
        check_roll_wrap(**kwargs)

        kwargs = {"data": data, "min_len": 1, "wrap": True, "only_nonzero": False}
        r = blocks(**kwargs)
        # should be one group
        assert len(r) == 2
        # should have every element
        check = set()
        for i in r:
            check.update(i)
        assert check == set(range(len(data)))
        check_roll_wrap(**kwargs)

        r = blocks(data, min_len=1, wrap=False, only_nonzero=False)
        assert len(r) == 3
        check = set()
        for i in r:
            check.update(i)
        assert check == set(range(len(data)))

        # CASE: blocks not at the end
        data = g.np.array([1, 0, 0, 0, 1, 1, 1, 0])
        kwargs = {"data": data, "min_len": 1, "wrap": True, "only_nonzero": True}
        r = blocks(**kwargs)
        assert len(r) == 2
        assert len(r[0]) == 1
        assert len(r[1]) == 3
        check_roll_wrap(**kwargs)

        # one block and one eligible but non-block point
        data = g.np.array([1, 0, 0, 0, 1, 1, 1, 1])
        r = blocks(data, min_len=2, wrap=True, only_nonzero=True)
        assert len(r) == 1
        assert g.np.allclose(data[r[0]], 1)

        # CASE: neither are in a block but together they are eligible
        data = g.np.array([1, 0, 0, 0, 1])
        kwargs = {"data": data, "min_len": 3, "wrap": True, "only_nonzero": True}
        r = blocks(**kwargs)
        assert len(r) == 0
        check_roll_wrap(**kwargs)
        kwargs["only_nonzero"] = False
        r = blocks(**kwargs)
        assert len(r) == 1
        assert g.np.allclose(data[r[0]], 0)
        check_roll_wrap(**kwargs)

        kwargs["data"] = g.np.abs(data - 1)
        # should be the same even inverted
        rn = blocks(**kwargs)
        assert len(r) == 1
        assert g.np.allclose(r[0], rn[0])
        check_roll_wrap(**kwargs)

        kwargs = {"data": data, "min_len": 2, "wrap": True, "only_nonzero": True}
        r = blocks(**kwargs)
        assert len(r) == 1
        assert set(r[0]) == {0, 4}
        check_roll_wrap(**kwargs)

    def test_runs(self):
        a = g.np.array(
            [-1, -1, -1, 0, 0, 1, 1, 2, 0, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
            dtype=g.np.int64,
        )
        r = g.trimesh.grouping.merge_runs(a)
        u = g.trimesh.grouping.unique_ordered(a)

        assert (g.np.diff(r) != 0).all()
        assert (g.np.diff(u) != 0).all()

        assert r.size == 12
        assert u.size == 11

    def test_cluster(self):
        # create some random points stacked with some zeros to cluster
        points = g.np.vstack(
            ((g.random((10000, 3)) * 5).astype(g.np.int64), g.np.zeros((100, 3)))
        )
        # should be at least one cluster
        assert len(g.trimesh.grouping.clusters(points, 0.01)) > 0
        # should be at least one group
        assert len(g.trimesh.grouping.group_distance(points, 0.01)) > 0

    def test_unique_float(self):
        a = g.np.arange(100) / 2.0
        t = g.np.tile(a, 2).flatten()

        unique = g.trimesh.grouping.unique_float(t)
        assert g.np.allclose(unique, a)

        unique, index, inverse = g.trimesh.grouping.unique_float(
            t, return_index=True, return_inverse=True
        )
        assert g.np.allclose(unique[inverse], t)
        assert g.np.allclose(unique, t[index])

    def test_group_rows(self):
        a = g.np.arange(100) / 2.0
        b = g.np.tile(a, 3).reshape((-1, 3))
        c = g.np.vstack((b, b))

        gr = g.trimesh.grouping.group_rows(c)
        assert g.np.shape(gr) == (100, 2)
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

        vec = g.np.column_stack((x, g.np.ones(len(x)), g.np.zeros(len(x))))
        vec = g.trimesh.unitize(vec)

        uv, ui = g.trimesh.grouping.group_vectors(vec)
        assert g.np.allclose(uv, vec)
        assert len(vec) == len(ui)
        assert g.np.allclose(uv[g.np.concatenate(ui)], vec)

        vec = g.np.vstack((vec, -vec))
        uv, ui = g.trimesh.grouping.group_vectors(vec)
        assert g.np.allclose(uv, vec)
        assert len(ui) == len(vec)

        uv, ui = g.trimesh.grouping.group_vectors(vec, include_negative=True)
        # since we included negative vectors, there should
        # be half the number of unique vectors and 2 indexes per vector
        assert g.np.shape(ui) == (100, 2)
        assert uv.shape == (100, 3)
        assert g.np.allclose(uv, vec[:100])

    def test_boolean_rows(self):
        a = g.np.arange(10).reshape((-1, 2))
        b = g.np.arange(10).reshape((-1, 2)) + 8
        # make one a different dtype
        b = b.astype(g.np.int32)

        # should have one overlapping row
        intersection = g.trimesh.grouping.boolean_rows(a, b, g.np.intersect1d)
        assert g.np.allclose(intersection.ravel(), [8, 9])

        diff = g.trimesh.grouping.boolean_rows(a, b, g.np.setdiff1d)
        assert g.np.allclose(g.np.unique(diff), g.np.arange(8))

    def test_broken(self):
        # create a broken mesh referencing
        # vertices that don't exist
        mesh = g.trimesh.Trimesh(vertices=[], faces=[[0, 1, 2]])
        # shouldn't error because there are no vertices
        # even though faces are wrong
        mesh.merge_vertices()

    def test_unique_ordered(self):
        a = g.np.array(
            [9, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 0, 2, 1, 1, 0, 0, -1, -1, -1],
            dtype=g.np.int64,
        )

        u, ind, inv = g.trimesh.grouping.unique_ordered(
            a, return_index=True, return_inverse=True
        )

        # indices are increasing, because we kept original order
        assert (g.np.diff(ind) > 0).all()
        # we can reconstruct original data
        assert (u[inv] == a).all()

    def test_unique_ordered_rows(self):
        # check the ordering of unique_rows
        v = g.random((100000, 3))
        v = g.np.vstack((v, v, v, v))

        # index, inverse
        i, iv = g.trimesh.grouping.unique_rows(v, keep_order=True)

        # get the unique values from the index
        u = v[i]

        # inverse of uniques should equal original array
        assert g.np.allclose(u[iv], v)
        # unique_ordered means indexes are in order
        assert (i == g.np.sort(i)).all()


def check_roll_wrap(**kwargs):
    """
    Check that blocks with wrapping enables returns the same
    value for all values of roll.

    Parameters
    ------------
    kwargs : dict
      Passed to trimesh.grouping.blocks
    """
    current = None
    # remove data from passed kwargs
    data = kwargs.pop("data")
    for i in range(len(data)):
        block = g.trimesh.grouping.blocks(g.np.roll(data, -i), **kwargs)
        # get result as a set of tuples with the rolling index
        # removed through a modulus, so we can compare equality
        check = {tuple(((j + i) % len(data)).tolist()) for j in block}
        if current is None:
            current = check
        # all values should be the same
        assert current == check


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
