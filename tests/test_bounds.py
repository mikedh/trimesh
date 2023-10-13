try:
    from . import generic as g
except BaseException:
    import generic as g


class BoundsTest(g.unittest.TestCase):
    def setUp(self):
        meshes = [g.get_mesh(i) for i in ["large_block.STL", "featuretype.STL"]]
        self.meshes = g.np.append(meshes, list(g.get_meshes(5)))

    def test_obb_mesh(self):
        """
        Test the OBB functionality in attributes of Trimesh objects
        """
        for m in self.meshes:
            g.log.info("Testing OBB of %s", m.metadata["file_name"])
            for i in range(6):
                # on the first run through don't transform the points to see
                # if we succeed in the meshes original orientation
                matrix = g.np.eye(4)
                if i > 0:
                    # get repeatable transforms
                    matrix = g.transforms[i]
                    m.apply_transform(matrix)

                box_ext = m.bounding_box_oriented.primitive.extents.copy()
                box_t = m.bounding_box_oriented.primitive.transform.copy()

                # determinant of rotation should be 1.0
                assert g.np.isclose(g.np.linalg.det(box_t[:3, :3]), 1.0)

                m.apply_transform(g.np.linalg.inv(box_t))

                test = m.bounds / (box_ext / 2.0)
                test_ok = g.np.allclose(test, [[-1, -1, -1], [1, 1, 1]])
                if not test_ok:
                    g.log.error("bounds test failed %s", str(test))
                assert test_ok

                m.apply_transform(matrix)
                m.apply_obb()

                # after applying the obb, the extents of the AABB
                # should be the same as the OBB
                close = g.np.allclose(
                    m.bounding_box.extents,
                    m.bounding_box_oriented.extents,
                    rtol=1e-3,
                    atol=1e-3,
                )
                if not close:
                    # m.visual.face_colors = [200, 0, 0, 100]
                    # (m + m.bounding_box_oriented).show()
                    # from IPython import embed
                    # embed()
                    raise ValueError(
                        "OBB extents incorrect:\n{}\n{}".format(
                            str(m.bounding_box.extents),
                            str(m.bounding_box_oriented.extents),
                        )
                    )

            c = m.bounding_cylinder  # NOQA
            s = m.bounding_sphere  # NOQA
            p = m.bounding_primitive  # NOQA

    def test_obb_points(self):
        """
        Test OBB functions with raw points as input
        """
        for dimension in [3, 2]:
            for _i in range(25):
                points = g.random((10, dimension))
                to_origin, extents = g.trimesh.bounds.oriented_bounds(points)

                assert g.trimesh.util.is_shape(to_origin, (dimension + 1, dimension + 1))
                assert g.trimesh.util.is_shape(extents, (dimension,))

                transformed = g.trimesh.transform_points(points, to_origin)

                transformed_bounds = [transformed.min(axis=0), transformed.max(axis=0)]

                for j in transformed_bounds:
                    # assert that the points once our obb to_origin transform is applied
                    # has a bounding box centered on the origin
                    assert g.np.allclose(g.np.abs(j), extents / 2.0)

                extents_tf = g.np.diff(transformed_bounds, axis=0).reshape(dimension)
                assert g.np.allclose(extents_tf, extents)

    def test_obb_coplanar_points(self):
        """
        Test OBB functions with coplanar points as input
        """
        for _i in range(5):
            points = g.np.zeros((5, 3))
            points[:, :2] = g.random((points.shape[0], 2))
            rot, _ = g.np.linalg.qr(g.random((3, 3)))
            points = g.np.matmul(points, rot)
            to_origin, extents = g.trimesh.bounds.oriented_bounds(points)

            assert g.trimesh.util.is_shape(to_origin, (4, 4))
            assert g.trimesh.util.is_shape(extents, (3,))

            transformed = g.trimesh.transform_points(points, to_origin)

            transformed_bounds = [transformed.min(axis=0), transformed.max(axis=0)]

            for j in transformed_bounds:
                # assert that the points once our obb to_origin transform is applied
                # has a bounding box centered on the origin
                assert g.np.allclose(g.np.abs(j), extents / 2.0)

            extents_tf = g.np.diff(transformed_bounds, axis=0).reshape(3)
            assert g.np.allclose(extents_tf, extents)

    def test_2D(self):
        for theta in g.np.linspace(0, g.np.pi * 2, 2000):
            # create some random rectangular-ish 2D points
            points = g.random((10, 2)) * [5, 1]

            # save the basic AABB of the points before rotation
            rectangle_pre = points.ptp(axis=0)

            # rotate them by an increment
            TR = g.trimesh.transformations.planar_matrix(theta=theta)
            points = g.trimesh.transform_points(points, TR)

            # find the OBB of the points
            T, rectangle = g.trimesh.bounds.oriented_bounds_2D(points)

            # apply the calculated OBB
            oriented = g.trimesh.transform_points(points, T)

            origin = oriented.min(axis=0) + oriented.ptp(axis=0) / 2.0

            # check to make sure the returned rectangle size is right
            assert g.np.allclose(oriented.ptp(axis=0), rectangle)
            # check to make sure the OBB consistently returns the
            # long axis in the same direction
            assert rectangle[0] > rectangle[1]
            # check to make sure result is actually returning an OBB
            assert g.np.allclose(origin, 0.0)
            # make sure OBB has less or same area as naive AABB
            assert g.np.prod(rectangle) <= g.np.prod(rectangle_pre)

    def test_cylinder(self):
        """
        Check bounding cylinders on basically a cuboid
        """
        # not rotationally symmetric
        mesh = g.get_mesh("featuretype.STL")

        height = 10.0
        radius = 1.0

        # spherical coordinates to loop through
        sphere = g.trimesh.util.grid_linspace([[0, 0], [g.np.pi * 2, g.np.pi * 2]], 5)

        for s in sphere:
            T = g.trimesh.transformations.spherical_matrix(*s)
            p = g.trimesh.creation.cylinder(radius=radius, height=height, transform=T)
            assert g.np.isclose(radius, p.bounding_cylinder.primitive.radius, rtol=0.01)
            assert g.np.isclose(height, p.bounding_cylinder.primitive.height, rtol=0.01)

            # regular mesh should have the same bounding cylinder
            # regardless of transform
            copied = mesh.copy()
            copied.apply_transform(T)
            assert g.np.isclose(
                mesh.bounding_cylinder.volume, copied.bounding_cylinder.volume, rtol=0.05
            )

    def test_random_cylinder(self):
        """
        Check exact cylinders with the bounding cylinder function.
        """
        for _i in range(20):
            # create a random cylinder
            c = g.trimesh.creation.cylinder(radius=1.0, height=10).permutate.transform()
            # bounding primitive should have same height and radius
            assert g.np.isclose(c.bounding_cylinder.primitive.height, 10, rtol=1e-6)
            assert g.np.isclose(c.bounding_cylinder.primitive.radius, 1, rtol=1e-6)
            # mesh is a cylinder, so center mass of bounding cylinder
            # should be exactly the same as the mesh center mass
            assert g.np.allclose(
                c.center_mass, c.bounding_cylinder.center_mass, rtol=1e-6
            )

    def test_bounding_egg(self):
        # create a distorted sphere mesh
        # center mass will be offset along Z
        i = g.trimesh.creation.icosphere()
        mask = i.vertices[:, 2] > 0.0
        i.vertices[:, 2][mask] *= 4.0

        # get a copy with a random transform
        p = i.permutate.transform()
        assert p.symmetry == "radial"

        # find the bounding cylinder with this random transform
        r = p.bounding_cylinder

        # transformed height should match source mesh
        assert g.np.isclose(i.vertices[:, 2].ptp(), r.primitive.height, rtol=1e-6)
        # slightly inflated cylinder should contain all
        # vertices of the source mesh
        assert r.buffer(0.01).contains(p.vertices).all()

    def test_obb_order(self):
        # make sure our sorting and transform flipping of
        # OBB extents are working by checking against a box

        extents = [10, 2, 3.5]
        extents_ordered = g.np.sort(extents)

        for _i in range(100):
            # transform box randomly in rotation and translation
            mat = g.trimesh.transformations.random_rotation_matrix()
            # translate in box -100 : +100
            mat[:3, 3] = (g.random(3) - 0.5) * 200

            # source mesh to check
            b = g.trimesh.creation.box(extents=extents, transform=mat)

            # calculated OBB primitive
            obb = b.bounding_box_oriented

            # make sure extents returned were ordered
            assert g.np.allclose(obb.primitive.extents, extents_ordered)

            # make sure mesh isn't reversing windings
            assert g.np.isclose(obb.to_mesh().volume, g.np.prod(extents))

            # make sure OBB has the same bounds as the source mesh
            # since it is a box the AABB of the OBB should be
            # the same as the AABB of the source mesh (lol)
            assert g.np.allclose(obb.bounds, b.bounds)

            # unordered extents and transforms
            transform, extents = g.trimesh.bounds.oriented_bounds(b, ordered=False)
            assert g.np.allclose(g.np.sort(extents), extents_ordered)
            # create a box from the unordered OBB information
            box = g.trimesh.creation.box(
                extents=extents, transform=g.np.linalg.inv(transform)
            )
            # make sure it is a real OBB too
            assert g.np.allclose(box.bounds, b.bounds)

    def test_bounds_tree(self):
        # test r-tree intersections
        for dimension in (2, 3):
            # create some (n, 2, 3) bounds
            bounds = g.np.array(
                [
                    [i.min(axis=0), i.max(axis=0)]
                    for i in [g.random((4, dimension)) for i in range(10)]
                ]
            )
            tree = g.trimesh.util.bounds_tree(bounds)
            for i, b in enumerate(bounds):
                assert i in set(tree.intersection(b.ravel()))
            # construct tree with per-row bounds
            tree = g.trimesh.util.bounds_tree(bounds.reshape((-1, dimension * 2)))
            for i, b in enumerate(bounds):
                assert i in set(tree.intersection(b.ravel()))

    def test_obb_corpus(self):
        # get some sample watertight meshes with nonzero volume
        min_volume = 0.1
        meshes = list(
            g.get_meshes(split=True, min_volume=min_volume, only_watertight=True)
        )
        g.log.debug(f"loaded {len(meshes)} meshes")

        if g.PY3:
            # our models corpus should have 200+ models
            # some loaders aren't available for Python2
            assert len(meshes) > 200

        # triple check we have positive volume
        assert all(m.volume > min_volume for m in meshes)

        # compute the OBB for every mesh profiling
        with g.Profiler() as P:
            obb = [m.bounding_box_oriented for m in meshes]
        g.log.debug(P.output_text())

        # now loop through mesh-obb pairs and validate
        for m, o in zip(meshes, obb):
            # move the mesh into the OBB frame
            check = m.copy().apply_transform(g.np.linalg.inv(o.primitive.transform))
            # check the mesh bounds against the claimed OBB bounds
            half = o.primitive.extents / 2.0
            check_extents = g.np.array([-half, half])
            # check that the OBB does contain the mesh
            assert g.np.allclose(check.bounds, check_extents, rtol=1e-4)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
