try:
    from . import generic as g
except BaseException:
    import generic as g

TEST_DIM = (10000, 3)


class CacheTest(g.unittest.TestCase):

    def test_hash(self):
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

        m = g.get_mesh('featuretype.STL')
        # log result values
        g.log.info('\nResult\nMD5:\n{}\nCRC:\n{}\nXX:\n{}'.format(
            m.vertices.md5(),
            m.vertices.crc(),
            m.vertices.fast_hash()))

        # crc should always be faster than MD5
        g.log.info('\nTime\nMD5:\n{}\nCRC:\n{}\nXX:\n{}'.format(
            mt, ct, xt))

        # CRC should be faster than MD5
        # this is NOT true if you blindly call adler32
        # but our speed check on import should assure this
        assert ct < mt

        # xxhash should be faster than CRC and MD5
        # it is sometimes slower on Windows/Appveyor TODO: figure out why
        if g.trimesh.caching.hasX and g.platform.system() == 'Linux':
            assert xt < mt
            assert xt < ct

    def test_track(self):
        """
        Check to make sure our fancy caching system only changes
        hashes when data actually changes.
        """
        # generate test data and perform numpy operations
        a = g.trimesh.caching.tracked_array(
            g.np.random.random(TEST_DIM))
        modified = []

        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])
        a[0][0] = 10
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        a[0][0] += 0.1
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        a[:10] = g.np.fliplr(a[:10])
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        a[1] = 5
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])
        a[2:] = 2
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        # these operations altered data and
        # the hash SHOULD have changed
        modified = g.np.array(modified)
        assert (g.np.diff(modified, axis=0) != 0).all()

        # now do slice operations which don't alter data
        modified = []
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])
        b = a[[0, 1, 2]]  # NOQA
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])
        c = a[1:]  # NOQA
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])
        # double slice brah
        a = a[::-1][::-1]
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        # these operations should have been cosmetic and
        # the hash should NOT have changed
        modified = g.np.array(modified)
        assert (g.np.diff(modified, axis=0) == 0).all()

        # now change stuff and see if checksums change
        a = g.trimesh.caching.tracked_array([0, 0, 4])
        modified = []

        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])
        a += 10
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        # assign some new data
        a = g.trimesh.caching.tracked_array([.125, 115.32444, 4],
                                            dtype=g.np.float64)

        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        a += [10, 0, 0]
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        a *= 10
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        # itruediv rather than idiv
        a /= 2.0
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        # idiv
        a /= 2.123
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        a -= 1.0
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        # in place floor division :|
        a //= 2
        modified.append([int(a.md5(), 16),
                         a.crc(),
                         a.fast_hash()])

        # these operations altered data and
        # the hash SHOULD have changed
        modified = g.np.array(modified)
        assert (g.np.diff(modified, axis=0) != 0).all()

    def test_contiguous(self):
        a = g.np.random.random((100, 3))
        t = g.trimesh.caching.tracked_array(a)

        # hashing will fail on non- contiguous arrays
        # make sure our utility function has handled this properly
        # for both MD5 and CRC
        assert t.md5() != t[::-1].md5()
        assert t.crc() != t[::-1].crc()
        assert t.fast_hash() != t[::-1].fast_hash()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
