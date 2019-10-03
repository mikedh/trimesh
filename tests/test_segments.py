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
            s = g.np.random.random((100, 2, dimension))
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

        seg = g.np.column_stack((
            g.np.zeros((3, 3)),
            [[0, 1, 0], [0, 1, 0], [0, 2, 0]])).reshape((-1, 2, 3))
        L = segments.colinear_pairs(seg)
        assert len(L) == 3

        n = segments.colinear_pairs(seg, length=0.01)
        assert len(n) == 1

    def test_extrude(self):
        from trimesh.path.segments import extrude
        # hand tuned segments
        manual = g.np.column_stack((
            g.np.zeros((3, 2)),
            [[0, 1], [0, -1], [1, 2]])).reshape((-1, 2, 2))

        for seg in [manual, g.random((10, 2, 2))]:
            height = 1.22
            v, f = extrude(segments=seg, height=height)
            # load extrusion as mesh
            mesh = g.trimesh.Trimesh(vertices=v, faces=f)
            # load line segments as path
            path = g.trimesh.load_path(seg)
            # compare area of mesh with source path
            assert g.np.isclose(mesh.area, path.length * height)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
