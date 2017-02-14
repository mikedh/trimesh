import generic as g


class GroupTests(g.unittest.TestCase):

    def test_unique_rows(self):
        count = 100
        subset = int(count/10)
        
        data = g.np.arange(count*3).reshape((-1,3)).astype(g.np.float)
        data[:subset] = data[0]

        unique, inverse = g.trimesh.grouping.unique_rows(data)

        assert (inverse[:subset] == 0).all()
        assert len(unique) == count - subset + 1

    def test_blocks(self):
        blocks = g.trimesh.grouping.blocks

        count = 100
        subset = int(count/10)
        a = g.np.zeros(count, dtype=g.np.int)

        result = blocks(a, min_len=0, only_nonzero=False)
        assert len(result) == 1
        assert len(result[0]) == count

        result = blocks(a, min_len=0, only_nonzero=True)
        assert len(result) == 0

        result = blocks(a, min_len=count+1, only_nonzero=False)
        assert len(result) == 0

        result = blocks(a, max_len=count-1, only_nonzero=False)
        assert len(result) == 0

        result = blocks(a, max_len=count+1, only_nonzero=False)
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
        
        
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
