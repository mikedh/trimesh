try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import triangle  # NOQA

    has_triangle = True
except ImportError:
    g.log.warning("No triangle! Not testing extrude primitives!")
    has_triangle = False


class ExtrudeTest(g.unittest.TestCase):
    def test_extrusion(self):
        if not has_triangle:
            return

        transform = g.trimesh.transformations.random_rotation_matrix()
        polygon = g.Point([0, 0]).buffer(0.5)
        e = g.trimesh.primitives.Extrusion(polygon=polygon, transform=transform)

        # will create an inflated version of the extrusion
        b = e.buffer(0.1)
        assert b.to_mesh().volume > e.to_mesh().volume
        assert b.contains(e.vertices).all()

        # try making it smaller
        b = e.buffer(-0.1)
        assert b.to_mesh().volume < e.to_mesh().volume
        assert e.contains(b.vertices).all()

        # try with negative height
        e = g.trimesh.primitives.Extrusion(
            polygon=polygon, height=-1.0, transform=transform
        )
        assert e.to_mesh().volume > 0.0

        # will create an inflated version of the extrusion
        b = e.buffer(0.1)
        assert b.to_mesh().volume > e.to_mesh().volume
        assert b.contains(e.vertices).all()

        # try making it smaller
        b = e.buffer(-0.1)
        assert b.to_mesh().volume < e.to_mesh().volume
        assert e.contains(b.vertices).all()

        # try with negative height and transform
        transform = [
            [1.0, 0.0, 0.0, -0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-0.0, -0.0, -1.0, -0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        e = g.trimesh.primitives.Extrusion(
            polygon=polygon, height=-1.0, transform=transform
        )
        assert e.to_mesh().volume > 0.0

        for T in g.transforms:
            current = e.copy().apply_transform(T)
            # get the special case OBB calculation for extrusions
            obb = current.bounding_box_oriented
            # check to make sure shortcutted OBB is the right size
            assert g.np.isclose(
                obb.volume, current.to_mesh().bounding_box_oriented.volume
            )
            # use OBB transform to project vertices of extrusion to plane
            points = g.trimesh.transform_points(
                current.vertices, g.np.linalg.inv(obb.primitive.transform)
            )
            # half extents of calculated oriented bounding box
            half = (g.np.abs(obb.primitive.extents) / 2.0) + 1e-3
            # every vertex should be inside OBB
            assert (points > -half).all()
            assert (points < half).all()

            assert current.direction.shape == (3,)
            assert current.origin.shape == (3,)

    def test_extrude_degen(self):
        # don't error on degenerate triangle
        # validate=True should remove the degenerate triangle
        # on creation of the mesh
        coords = g.np.array([[0, 0], [0, 0], [0, 1], [1, 0]])
        mesh = g.trimesh.creation.extrude_triangulation(
            vertices=coords, faces=[[0, 1, 2], [0, 2, 3]], height=1, validate=True
        )
        assert mesh.is_watertight


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
