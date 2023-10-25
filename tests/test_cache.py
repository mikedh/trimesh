try:
    from . import generic as g
except BaseException:
    import generic as g

TEST_DIM = (100000, 3)


class CacheTest(g.unittest.TestCase):
    def test_track(self):
        """
        Check to make sure our fancy caching system only changes
        hashes when data actually changes.
        """

        original = g.trimesh.caching.hash_fast
        options = [
            g.trimesh.caching.hash_fast,
            g.trimesh.caching.hash_fallback,
            g.trimesh.caching.sha256,
        ]

        for option in options:
            g.log.info(f"testing hash function: {option.__name__}")
            g.trimesh.caching.hash_fast = option

            # generate test data and perform numpy operations
            a = g.trimesh.caching.tracked_array(g.random(TEST_DIM))
            modified = [hash(a)]
            a[0][0] = 10
            modified.append(hash(a))
            a[0][0] += 0.1
            modified.append(hash(a))
            a[:10] = g.np.fliplr(a[:10])
            modified.append(hash(a))
            a[1] = 5
            modified.append(hash(a))
            a[2:] = 2
            modified.append(hash(a))
            # these operations altered data and
            # the hash SHOULD have changed
            modified = g.np.array(modified, dtype=g.np.int64)
            assert (g.np.diff(modified) != 0).all()

            # now do slice operations which don't alter data
            modified = []
            modified.append(hash(a))
            b = a[[0, 1, 2]]  # NOQA
            modified.append(hash(a))
            c = a[1:]  # NOQA
            modified.append(hash(a))
            # double slice brah
            a = a[::-1][::-1]
            modified.append(hash(a))
            # these operations should have been cosmetic and
            # the hash should NOT have changed
            modified = g.np.array(modified, dtype=g.np.int64)
            assert (g.np.diff(modified, axis=0) == 0).all()

            # now change stuff and see if checksums change
            a = g.trimesh.caching.tracked_array([0, 0, 4])
            modified = []

            modified.append(hash(a))
            a += 10
            modified.append(hash(a))
            # assign some new data
            a = g.trimesh.caching.tracked_array([0.125, 115.32444, 4], dtype=g.np.float64)

            modified.append(hash(a))
            a += [10, 0, 0]
            modified.append(hash(a))
            a *= 10
            modified.append(hash(a))
            # itruediv rather than idiv
            a /= 2.0
            modified.append(hash(a))
            # idiv
            a /= 2.123
            modified.append(hash(a))
            a -= 1.0
            modified.append(hash(a))
            # in place floor division :|
            a //= 2
            modified.append(hash(a))
            # in-place sort
            a.sort()
            modified.append(hash(a))

            # these operations altered data and
            # the hash SHOULD have changed
            modified = g.np.array(modified, dtype=g.np.int64)
            assert (g.np.diff(modified) != 0).all()

        # reset our patched hash function
        g.trimesh.caching.hash_fast = original

    def test_contiguous(self):
        a = g.random((100, 3))
        t = g.trimesh.caching.tracked_array(a)

        original = g.trimesh.caching.hash_fast
        options = [
            g.trimesh.caching.hash_fast,
            g.trimesh.caching.hash_fallback,
            g.trimesh.caching.sha256,
        ]

        for option in options:
            g.log.info(f"testing hash function: {option.__name__}")
            g.trimesh.caching.hash_fast = option
            # hashing will fail on non- contiguous arrays
            # make sure our utility function has handled this
            assert hash(t) != hash(t[::-1])

        # reset our patched hash function
        g.trimesh.caching.hash_fast = original

    def test_mutable(self):
        """
        Run some simple tests on mutable DataStore objects
        """
        d = g.trimesh.caching.DataStore()

        d["hi"] = g.random(100)
        hash_initial = hash(d)
        # mutate internal data
        d["hi"][0] += 1
        assert hash(d) != hash_initial

        # should be mutable by default
        assert d.mutable
        # set data to immutable
        d.mutable = False

        try:
            d["hi"][1] += 1
        except ValueError:
            # should be raised when array is marked as read only
            return
        # we shouldn't make it past the try-except
        raise ValueError("mutating data worked when it shouldn't!")

    def test_transform(self):
        """
        apply_transform tries to not dump the full cache
        """
        m = g.get_mesh("featuretype.STL")
        # should populate edges_face
        e_len = len(m.edges)
        # should maintain required properties
        m.apply_transform(g.transforms[1])
        # should still be in the cache
        assert len(m.edges_face) == e_len

    def test_simple_collision(self):
        faces1 = g.np.array(
            [
                0,
                1,
                2,
                0,
                3,
                1,
                0,
                2,
                4,
                0,
                4,
                5,
                5,
                6,
                3,
                5,
                3,
                0,
                7,
                1,
                3,
                7,
                3,
                6,
                4,
                2,
                1,
                4,
                1,
                7,
                5,
                4,
                7,
                6,
                5,
                7,
            ],
            dtype=g.np.int64,
        ).reshape(-1, 3)
        faces2 = g.np.array(
            [
                0,
                1,
                2,
                0,
                3,
                1,
                2,
                4,
                0,
                5,
                4,
                2,
                6,
                3,
                0,
                6,
                0,
                4,
                6,
                1,
                3,
                6,
                7,
                1,
                2,
                7,
                5,
                2,
                1,
                7,
                4,
                5,
                7,
                6,
                4,
                7,
            ],
            dtype=g.np.int64,
        ).reshape(-1, 3)
        hash_fast = g.trimesh.caching.hash_fast
        assert hash_fast(faces1) != hash_fast(faces2)

    def test_scalar(self):
        # check to make sure we're doing the same
        # thing as numpy with our subclass
        o = g.np.arange(10)
        a = g.trimesh.caching.tracked_array(o)
        # tracked array should be different type
        # but with the same values (i.e. subclass)
        assert not isinstance(o, type(a))
        assert isinstance(a, g.np.ndarray)
        assert g.np.allclose(a, o)

        # this should be a scalar and subclasses
        # for some reason return a different type?
        assert isinstance(a.max(), type(o.max()))

    def test_method_combinations(self):
        # run combinatorial guesses of arguments on every method
        # of the subclassed `ndarray` and see if we can get trackedarray
        # to return incorrect hashes. Hopefully this catches new methods

        # only run this test on python 3+ as they changed tobytes
        if not g.PY3:
            return

        import itertools
        import warnings

        import numpy as np

        from trimesh.caching import tracked_array

        dim = (100, 3)

        # generate a bunch of arguments for every function of an `ndarray` so
        # we can see if the functions mutate
        flat = [
            2.3,
            1,
            10,
            4.2,
            [3, -1],
            {"shape": 10},
            np.int64,
            np.float64,
            True,
            True,
            False,
            False,
            g.random(dim),
            g.random(dim[::1]),
            "shape",
        ]

        # start with no arguments
        attempts = [()]
        # add a single argument from our guesses
        attempts.extend([(A,) for A in flat])
        # add 2 and 3 length permutations of our guesses
        attempts.extend([tuple(G) for G in itertools.product(flat, repeat=2)])

        # adding 3-length permutations makes this test 10x slower but if you
        # are suspicious of a method caching you could uncomment this out:
        # attempts.extend([tuple(G) for G in itertools.permutations(flat, 3)])
        skip = set()

        # collect functions which mutate arrays but don't change our hash
        broken = []

        with warnings.catch_warnings():
            # ignore all warnings inside this context manager
            warnings.filterwarnings("ignore")

            for method in list(dir(tracked_array(g.random(dim)))):
                if method in skip:
                    continue

                failures = []
                g.log.debug(f"hash check: `{method}`")
                for A in attempts:
                    m = g.random((100, 3))
                    true_pre = m.tobytes()
                    m = tracked_array(m)
                    hash_pre = hash(m)
                    try:
                        eval(f"m.{method}(*A)")
                    except BaseException as J:
                        failures.append(str(J))

                    hash_post = hash(m)
                    true_post = m.tobytes()

                    # if tobytes disagrees with our hashing logic
                    # it indicates we have cached incorrectly
                    if (hash_pre == hash_post) != (true_pre == true_post):
                        broken.append((method, A))

        if len(broken) > 0:
            method_busted = {method for method, _ in broken}
            raise ValueError(
                f"`TrackedArray` incorrectly hashing methods: {method_busted}"
            )

    def test_validate(self):
        # create a mesh with two duplicate triangles
        # and one degenerate triangle
        m = g.trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            faces=[[3, 4, 4], [0, 1, 2], [0, 1, 2]],
            validate=False,
        )

        # should not have removed any triangles
        assert m.triangles.shape == (3, 3, 3)
        # should remove every triangle except one
        m.process(validate=True)
        assert m.triangles.shape == (1, 3, 3)

    def test_smooth_shade(self, count=10):
        # test to make sure the smooth shaded copy is cached correctly
        mesh = g.trimesh.creation.cylinder(radius=1, height=10)
        scene = g.trimesh.Scene({"mesh": mesh})

        initial = scene.camera_transform.copy()

        hashes = set()
        for n in range(count):
            angle = g.np.pi * n / count
            matrix = g.trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
            scene.geometry["mesh"].apply_transform(matrix)
            hashes.add(hash(scene.geometry["mesh"].smooth_shaded))
            scene.camera_transform = initial

        # the smooth shade should be unique for every transform
        assert len(hashes) == count


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
