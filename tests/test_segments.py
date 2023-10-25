try:
    from . import generic as g
except BaseException:
    import generic as g


class SegmentsTest(g.unittest.TestCase):
    def test_param(self):
        from trimesh.path import segments

        # check 2D and 3D
        for dimension in [2, 3]:
            # a bunch of random line segments
            s = g.random((100, 2, dimension))
            # convert segment to point on line closest to origin
            # as well as a vector and two distances along vector
            param = segments.segments_to_parameters(s)
            # convert parameterized back to segments
            roundtrip = segments.parameters_to_segments(*param)
            # they should be the same after a roundtrip
            assert g.np.allclose(s, roundtrip)

            # make index 1 the first segment but offset along vector
            # IE make s[0] colinear with s[1]
            s[1] = s[0] + 10 * (s[0][0] - s[0][1])
            # calculate colinear pairs
            colinear = segments.colinear_pairs(s)

            # due to our wangling the first and second index
            # should be a colinear pair
            assert {0, 1} in [set(i) for i in colinear]

    def test_colinear(self):
        from trimesh.path import segments

        seg = g.np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.5, 0.0], [0.0, 0.75, 0.0]],
                [[0.0, 2.1, 0.0], [0.0, 2.2, 0.0]],
                [[0.0, 2.0, 0.0], [0.0, 2.3, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )

        # get the unit direction vector for the segments
        unit = g.trimesh.unitize(g.np.diff(seg, axis=1).reshape((-1, 3)))

        L = segments.colinear_pairs(seg[:3])
        assert len(L) == 3
        # make sure all pairs are really colinear
        dots = [g.np.dot(*row) for row in unit[L]]
        assert (g.np.isclose(dots, 1.0) | g.np.isclose(dots, -1)).all()

        L = segments.colinear_pairs(seg)
        dots = [g.np.dot(*row) for row in unit[L]]
        assert (g.np.isclose(dots, 1.0) | g.np.isclose(dots, -1)).all()

        epsilon = 1e-6
        # length should only include vectors with one
        # vertex closer than epsilon
        n = segments.colinear_pairs(seg, length=epsilon)
        dots = [g.np.dot(*row) for row in unit[L]]
        assert (g.np.isclose(dots, 1.0) | g.np.isclose(dots, -1)).all()

        for pair in n:
            val = seg[pair]
            close = g.np.append(
                (val[0] - val[1]).ptp(axis=1), (val[0] - val[1][::-1]).ptp(axis=1)
            ).min()
            assert close < epsilon

    def test_extrude(self):
        from trimesh.path.segments import extrude

        # hand tuned segments
        manual = g.np.column_stack(
            (g.np.zeros((3, 2)), [[0, 1], [0, -1], [1, 2]])
        ).reshape((-1, 2, 2))

        for seg in [manual, g.random((10, 2, 2))]:
            height = 1.22
            v, f = extrude(segments=seg, height=height)
            # load extrusion as mesh
            mesh = g.trimesh.Trimesh(vertices=v, faces=f)
            # load line segments as path
            path = g.trimesh.load_path(seg)
            # compare area of mesh with source path
            assert g.np.isclose(mesh.area, path.length * height)

    def test_resample(self):
        from trimesh.path.segments import length, resample

        # create some random segments
        seg = g.random((1000, 2, 3))
        # set a maximum segment length
        maxlen = 0.1
        # one of the original segments should be longer than maxlen
        assert (length(seg, summed=False) > maxlen).any()
        # resample to be all shorter than maxlen
        res = resample(seg, maxlen=maxlen)
        # check lengths of the resampled result
        assert (length(res, summed=False) < maxlen).all()
        # make sure overall length hasn't changed
        assert g.np.isclose(length(res), length(seg))

        # now try with indexes returned
        res, index = resample(seg, maxlen=maxlen, return_index=True)
        # check lengths of the resampled result
        assert (length(res, summed=False) < maxlen).all()
        # make sure overall length hasn't changed
        assert g.np.isclose(length(res), length(seg))

    def test_clean(self):
        from trimesh.path.segments import clean, resample

        seg = g.np.array(
            [[[0, 0], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[0, 1], [0, 0]]],
            dtype=g.np.float64,
        )

        c = clean(seg)
        assert len(seg) == len(c)
        # bounding box should be the same
        assert g.np.allclose(
            c.reshape((-1, 2)).min(axis=0), seg.reshape((-1, 2)).min(axis=0)
        )
        assert g.np.allclose(
            c.reshape((-1, 2)).max(axis=0), seg.reshape((-1, 2)).max(axis=0)
        )

        # resample to shorten
        r = resample(seg, maxlen=0.3)
        assert r.shape == (16, 2, 2)
        # after cleaning should be back to 4 segments
        rc = clean(r)
        assert rc.shape == (4, 2, 2)

    def test_svg(self):
        from trimesh.path.segments import to_svg

        # create some 2D segments
        seg = g.random((1000, 2, 2))
        # make one of the segments a duplicate
        seg[0] = seg[-1]
        # create an SVG path string
        svg = to_svg(seg, merge=False)
        # should be one move and one line per segment
        assert svg.count("M") == len(seg)
        assert svg.count("L") == len(seg)

        # try with a transform
        svg = to_svg(seg, matrix=g.np.eye(3), merge=False)
        assert svg.count("M") == len(seg)
        assert svg.count("L") == len(seg)

        # remove the duplicate segments
        svg = to_svg(seg, matrix=g.np.eye(3), merge=True)
        assert svg.count("M") < len(seg)
        assert svg.count("L") < len(seg)

        try:
            to_svg(g.random((100, 2, 3)))
        except ValueError:
            return
        raise ValueError("to_svg accepted wrong input!")


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
