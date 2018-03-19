import generic as g

TEST_DIM = (10000,3)

class CacheTest(g.unittest.TestCase):

    def test_hash(self):

        m = g.get_mesh('featuretype.STL')

        md = m.vertices.md5()
        c = m.vertices.crc()
        x = m.vertices.fast_hash()

        
        
        setup = 'import numpy, trimesh;'
        setup += 'd = numpy.random.random((10000,3));'
        setup += 't = trimesh.caching.tracked_array(d)'

        count = 10000
        
        mt = g.timeit.timeit(setup=setup,
            stmt='t._modified_m=True;t.md5()',
            number=count)
        ct = g.timeit.timeit(setup=setup,
            stmt='t._modified_c=True;t.crc()',
                             number=count)
        xt = g.timeit.timeit(setup=setup,
            stmt='t._modified_x=True;t.fast_hash()',
            number=count)

        g.log.info('MD5:\n{}\nCRC:\n{}\nXX:\n{}'.format(
            md,
            c,
            x))
        
        # crc should always be faster than MD5's
        g.log.info('MD5:\n{}\nCRC:\n{}\nXX:\n{}'.format(
            mt,
            ct,
            xt))

        if g.trimesh.caching.hasX:
            assert xt < mt
            assert xt < ct

        
    def test_track(self):
        a = g.trimesh.caching.tracked_array(g.np.random.random(TEST_DIM))
        modified = g.collections.deque()
        modified.append(int(a.md5(), 16))
        a[0][0] = 10
        modified.append(int(a.md5(), 16))
        a[1] = 5
        modified.append(int(a.md5(), 16))
        a[2:] = 2
        modified.append(int(a.md5(), 16))
        self.assertTrue((g.np.diff(modified) != 0).all())

        modified = g.collections.deque()
        modified.append(int(a.md5(), 16))
        b = a[[0, 1, 2]]
        modified.append(int(a.md5(), 16))
        c = a[1:]
        modified.append(int(a.md5(), 16))
        self.assertTrue((g.np.diff(modified) == 0).all())

        a = g.trimesh.caching.tracked_array([0, 0, 4])
        pre_crc = a.crc()
        a += 10
        assert a.crc() != pre_crc

        a = g.trimesh.caching.tracked_array([.125, 115.32444, 4],
                                            dtype=g.np.float64)
        modified = g.collections.deque()
        modified.append(a.crc())
        a += [10, 0, 0]
        modified.append(a.crc())
        a *= 10
        # __idiv__ is weird, and not working
        # modified.append(a.crc())
        #a /= 2.0
        modified.append(a.crc())
        a -= 1.0
        modified.append(a.crc())
        a //= 2
        modified.append(a.crc())
        assert (g.np.diff(modified) != 0).all()

    def test_contiguous(self):
        a = g.np.random.random((100, 3))
        t = g.trimesh.caching.tracked_array(a)

        # hashing will fail on non- contiguous arrays
        # make sure our utility function has handled this properly
        # for both MD5 and CRC
        t[::-1].md5()
        t[::-1].crc()


        
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
