try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.triangles import barycentric_to_points, points_to_barycentric


class TrianglesTest(g.unittest.TestCase):
    def test_barycentric(self):
        for m in g.get_meshes(4):
            # a simple test which gets the barycentric coordinate at each of the three
            # vertices, checks to make sure the barycentric is [1,0,0] for the vertex
            # and then converts back to cartesian and makes sure the original points
            #  are the same as the conversion and back
            for method in ["cross", "cramer"]:
                for i in range(3):
                    barycentric = points_to_barycentric(
                        m.triangles, m.triangles[:, i], method=method
                    )
                    assert (
                        g.np.abs(barycentric - g.np.roll([1.0, 0, 0], i)) < 1e-8
                    ).all()

                    points = barycentric_to_points(m.triangles, barycentric)
                    assert (g.np.abs(points - m.triangles[:, i]) < 1e-8).all()

    def test_barycentric_2D(self):
        # points_to_barycentric should work in 2D and 3D
        tri = g.np.array([[[0, 0], [1, 0], [0, 1]]], dtype=g.np.float64)
        points = g.np.array([[0.25, 0.25]])
        b = points_to_barycentric(tri, points)
        assert ((b > 0) & (b < 1.0)).all()

        r = barycentric_to_points(tri, b)

        assert g.np.allclose(r, points)

    def test_closest(self):
        closest = g.trimesh.triangles.closest_point(
            triangles=g.data["triangles"]["triangles"],
            points=g.data["triangles"]["points"],
        )

        comparison = (closest - g.data["triangles"]["closest"]).all()

        assert (comparison < 1e-8).all()
        g.log.info("finished closest check on %d triangles", len(closest))

    def test_closest_obtuse(self):
        # simple triangle in the xy-plane with an obtuse corner at vertex A
        ABC = g.np.float32([[0, 0, 0], [2, 0, 0], [-2, 1, 0]])
        D = g.np.float32([1, -1, 0])

        # ground truth: closest point from D is the center of the AB edge:
        # (1,0,0)
        gt_closest = g.np.float32([1, 0, 0])
        tm_closest = g.trimesh.triangles.closest_point([ABC], [D])[0]

        assert g.np.linalg.norm(gt_closest - tm_closest) < g.tol.merge

        # create a circle of points around the triangle
        # with a radius so that all points are outside of the triangle
        radius = 3
        nPtsOnCircle = 100
        alphas = g.np.linspace(
            g.np.pi / nPtsOnCircle, g.np.pi * 2 - g.np.pi / nPtsOnCircle, nPtsOnCircle
        )
        ptsOnCircle = (
            g.np.transpose([g.np.cos(alphas), g.np.sin(alphas), g.np.zeros(nPtsOnCircle)])
            * radius
        )

        def norm(v):
            return g.np.linalg.norm(v, axis=-1)

        def distToLine(o, v, p):
            return norm((o - p) - g.np.dot(o - p, v) * v)

        def distPointToEdge(U, V, P):  # edge [U, V], point P
            UtoV = V - U
            UtoP = P - U
            VtoP = P - V

            if g.np.dot(UtoV, UtoP) <= 0:
                # P is 'behind' U
                return norm(UtoP)
            elif g.np.dot(-UtoV, VtoP) <= 0:
                # P is 'behind' V
                return norm(VtoP)
            else:
                # P is 'between' U and V
                return distToLine(U, UtoV / norm(UtoV), P)

        # get closest points from trimesh and compute distances to the circle
        # points
        tm_dists = norm(
            ptsOnCircle
            - g.trimesh.triangles.closest_point([ABC] * nPtsOnCircle, ptsOnCircle)
        )
        # compute naive point-to-edge distances for all points and take the min of
        # the three edges
        gt_dists = g.np.float32(
            [
                [distPointToEdge(ABC[i], ABC[(i + 1) % 3], pt) for i in range(3)]
                for pt in ptsOnCircle
            ]
        ).min(axis=1)
        diff_dists = tm_dists - gt_dists

        assert g.np.dot(diff_dists, diff_dists) < g.tol.merge

    def test_closest_scale(self):
        # the closest point must not depend on the units of the triangles
        rng = g.np.random.default_rng(0)
        triangles = rng.random((500, 3, 3))
        # points on each triangle: random ones and a dense grid of samples
        inside = g.np.einsum("ij,ijk->ik", rng.dirichlet([1, 1, 1], 500), triangles)
        weights = rng.dirichlet([1, 1, 1], 200)
        samples = g.np.einsum("sj,ijk->isk", weights, triangles)
        outside = rng.random((500, 3)) * 3 - 1

        for scale in 10.0 ** g.np.arange(-8, 9, 2):
            # a point on a triangle is its own closest point
            closest = g.trimesh.triangles.closest_point(triangles * scale, inside * scale)
            assert g.np.abs(closest / scale - inside).max() < 1e-10

            closest = (
                g.trimesh.triangles.closest_point(triangles * scale, outside * scale)
                / scale
            )
            # the result has to be on the triangle
            barycentric = g.trimesh.triangles.points_to_barycentric(triangles, closest)
            assert barycentric.min() > -1e-10
            assert g.np.abs(barycentric.sum(axis=1) - 1).max() < 1e-10
            on_plane = g.np.einsum("ij,ijk->ik", barycentric, triangles)
            assert g.np.abs(on_plane - closest).max() < 1e-10
            # and no other point of the triangle can be closer
            distance = g.np.linalg.norm(closest - outside, axis=1)
            sampled = g.np.linalg.norm(samples - outside[:, None, :], axis=2)
            assert (distance <= sampled.min(axis=1) + 1e-10).all()

    def test_closest_collinear(self):
        # collinear triangles in general position carry rounding noise
        rng = g.np.random.default_rng(1)
        a, b = rng.random((2, 20000, 3))
        factor = rng.random((len(a), 1)) * 3 - 1
        triangles = g.np.stack([a, b, a + factor * (b - a)], axis=1)
        points = rng.random((len(a), 3)) * 3 - 1

        # the truth is the closest point on the longest of the three segments
        truth = g.np.full(len(a), g.np.inf)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            origin, vector = triangles[:, i], triangles[:, j] - triangles[:, i]
            weight = ((points - origin) * vector).sum(axis=1) / (vector**2).sum(axis=1)
            on_segment = origin + g.np.clip(weight, 0, 1)[:, None] * vector
            truth = g.np.minimum(truth, g.np.linalg.norm(on_segment - points, axis=1))

        for scale in (1e-4, 1.0, 1e4):
            closest = g.trimesh.triangles.closest_point(triangles * scale, points * scale)
            distance = g.np.linalg.norm(closest / scale - points, axis=1)
            assert g.np.abs(distance - truth).max() < 1e-9

    def test_closest_degenerate(self):
        # Degenerate faces reduce to a segment or a single point.
        triangles = g.np.array(
            [
                [[0, 0, 0], [0, 0, 0], [2, 0, 0]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                [[0, 0, 0], [0, 0, 0], [0, 2, 0]],
                [[0, 0, 0], [2, 0, 0], [2, 0, 0]],
                [[0, 0, 0], [2, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
            ],
            dtype=g.np.float64,
        )
        points = g.np.array(
            [
                [1, 1, 0],
                [5, 2, 3],
                [0, 3, 1],
                [1, 1, 0],
                [1, 1, 0],
                [1.5, 1, 0],
                [0.5, 0.5, 1],
            ]
        )
        expected = g.np.array(
            [
                [1, 0, 0],
                [1, 2, 3],
                [0, 2, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1.5, 0, 0],
                [0.5, 0.5, 0],
            ]
        )

        with g.np.errstate(divide="raise", invalid="raise"):
            closest = g.trimesh.triangles.closest_point(triangles, points)
        g.np.testing.assert_allclose(closest, expected)

    def test_degenerate(self):
        tri = [
            [[0, 0, 0], [1, 0, 0], [-0.5, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [10, 10, 0]],
            [[0, 0, 0], [0, 0, 2], [0, 0, 2.2]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ]

        tri_gt = [False, False, False, True]

        r = g.trimesh.triangles.nondegenerate(tri)
        assert len(r) == len(tri)
        assert (r == tri_gt).all()

    def test_angles(self):
        # a zero- area triangle
        tris = g.np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 0]]], dtype=g.np.float64)
        angles = g.trimesh.triangles.angles(tris)
        # degenerate angles should be zero, not NaN
        g.log.debug(angles)
        assert g.np.allclose(angles, 0.0)

        # an equilateral triangle
        tris = g.np.array(
            [[[-1, 0, 0], [1, 0, 0], [0, g.np.sqrt(3), 0]]], dtype=g.np.float64
        )
        angles = g.trimesh.triangles.angles(tris)
        # degenerate angles should be zero, not NaN
        assert g.np.allclose(angles, g.np.radians(60))

        # an equilateral triangle transformed into space
        tris = g.trimesh.transform_points(
            g.np.array([[-1, 0, 0], [1, 0, 0], [0, g.np.sqrt(3), 0]], dtype=g.np.float64),
            next(g.random_transforms(1, translate=0.0)),
        ).reshape((-1, 3, 3))
        angles = g.trimesh.triangles.angles(tris)
        # all angles should be 60 degrees
        assert g.np.allclose(angles, g.np.radians(60))

        # an 3-4-5 right triangle
        tris = g.trimesh.transform_points(
            g.np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=g.np.float64),
            next(g.random_transforms(1, translate=0.0)),
        ).reshape((-1, 3, 3))
        # get angles
        angles = g.trimesh.triangles.angles(tris)
        # make sure they match a 3-4-5
        assert g.np.allclose(
            g.np.sort(angles.ravel()),
            [g.np.arcsin(3.0 / 5), g.np.arcsin(4.0 / 5), g.np.pi / 2],
        )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
