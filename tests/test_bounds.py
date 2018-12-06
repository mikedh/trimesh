try:
    from . import generic as g
except BaseException:
    import generic as g


class BoundsTest(g.unittest.TestCase):

    def setUp(self):
        meshes = [g.get_mesh(i) for i in ['large_block.STL',
                                          'featuretype.STL']]
        self.meshes = g.np.append(meshes, list(g.get_meshes(5)))

    def test_obb_mesh(self):
        """
        Test the OBB functionality in attributes of Trimesh objects
        """
        for m in self.meshes:
            g.log.info('Testing OBB of %s', m.metadata['file_name'])
            for i in range(6):
                # on the first run through don't transform the points to see
                # if we succeed in the meshes original orientation
                matrix = g.np.eye(4)
                if i > 0:
                    matrix = g.trimesh.transformations.random_rotation_matrix()
                    matrix[0:3, 3] = (g.np.random.random(3) - .5) * 100
                    m.apply_transform(matrix)

                box_ext = m.bounding_box_oriented.primitive.extents.copy()
                box_t = m.bounding_box_oriented.primitive.transform.copy()

                m.apply_transform(g.np.linalg.inv(box_t))

                test = m.bounds / (box_ext / 2.0)
                test_ok = g.np.allclose(test, [[-1, -1, -1], [1, 1, 1]])
                if not test_ok:
                    g.log.error('bounds test failed %s',
                                str(test))
                assert test_ok

                m.apply_transform(matrix)
                m.apply_obb()

                # after applying the obb, the extents of the AABB
                # should be the same as the OBB
                close = g.np.allclose(m.bounding_box.extents,
                                      m.bounding_box_oriented.extents,
                                      rtol=1e-3,
                                      atol=1e-3)
                if not close:
                    #m.visual.face_colors = [200, 0, 0, 100]
                    #(m + m.bounding_box_oriented).show()
                    #from IPython import embed
                    # embed()
                    raise ValueError('OBB extents incorrect:\n{}\n{}'.format(
                        str(m.bounding_box.extents),
                        str(m.bounding_box_oriented.extents)))

            c = m.bounding_cylinder
            s = m.bounding_sphere
            p = m.bounding_primitive

    def test_obb_points(self):
        """
        Test OBB functions with raw points as input
        """
        for dimension in [3, 2]:
            for i in range(25):
                points = g.np.random.random((10, dimension))
                to_origin, extents = g.trimesh.bounds.oriented_bounds(points)

                assert g.trimesh.util.is_shape(to_origin,
                                               (dimension + 1, dimension + 1))
                assert g.trimesh.util.is_shape(extents, (dimension,))

                transformed = g.trimesh.transform_points(points, to_origin)

                transformed_bounds = [transformed.min(axis=0),
                                      transformed.max(axis=0)]

                for i in transformed_bounds:
                    # assert that the points once our obb to_origin transform is applied
                    # has a bounding box centered on the origin
                    assert g.np.allclose(g.np.abs(i), extents / 2.0)

                extents_tf = g.np.diff(
                    transformed_bounds, axis=0).reshape(dimension)
                assert g.np.allclose(extents_tf, extents)

    def test_2D(self):
        for theta in g.np.linspace(0, g.np.pi * 2, 2000):
            # create some random rectangular-ish 2D points
            points = g.np.random.random((10, 2)) * [5, 1]

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
            assert g.np.product(rectangle) <= g.np.product(rectangle_pre)

    def test_cylinder(self):
        """
        """
        # not rotationally symmetric
        mesh = g.get_mesh('featuretype.STL')

        height = 10.0
        radius = 1.0

        # spherical coordinates to loop through
        sphere = g.trimesh.util.grid_linspace([[0, 0],
                                               [g.np.pi * 2, g.np.pi * 2]],
                                              5)
        for s in sphere:
            T = g.trimesh.transformations.spherical_matrix(*s)
            p = g.trimesh.creation.cylinder(radius=radius,
                                            height=height,
                                            transform=T)
            assert g.np.isclose(radius,
                                p.bounding_cylinder.primitive.radius,
                                rtol=.01)
            assert g.np.isclose(height,
                                p.bounding_cylinder.primitive.height,
                                rtol=.01)

            # regular mesh should have the same bounding cylinder
            # regardless of transform
            copied = mesh.copy()
            copied.apply_transform(T)
            assert g.np.isclose(mesh.bounding_cylinder.volume,
                                copied.bounding_cylinder.volume,
                                rtol=.05)

    def test_obb_order(self):
        # make sure our sorting and transform flipping of
        # OBB extents are working by checking against a box

        extents = [10, 2, 3.5]
        extents_ordered = g.np.sort(extents)

        for i in range(100):
            # transform box randomly in rotation and translation
            mat = g.trimesh.transformations.random_rotation_matrix()
            # translate in box -100 : +100
            mat[:3, 3] = (g.np.random.random(3) - .5) * 200

            # source mesh to check
            b = g.trimesh.creation.box(extents=extents,
                                       transform=mat)

            # calculated OBB primitive
            obb = b.bounding_box_oriented

            # make sure extents returned were ordered
            assert g.np.allclose(obb.primitive.extents,
                                 extents_ordered)

            # make sure mesh isn't reversing windings
            assert g.np.isclose(obb.to_mesh().volume,
                                g.np.product(extents))

            # make sure OBB has the same bounds as the source mesh
            # since it is a box the AABB of the OBB should be
            # the same as the AABB of the source mesh (lol)
            assert g.np.allclose(obb.bounds,
                                 b.bounds)

            # unordered extents and transforms
            uT, uE = g.trimesh.bounds.oriented_bounds(b, ordered=False)
            assert g.np.allclose(g.np.sort(uE), extents_ordered)
            # create a box from the unordered OBB information
            uB = g.trimesh.creation.box(
                extents=uE, transform=g.np.linalg.inv(uT))
            # make sure it is a real OBB too
            assert g.np.allclose(uB.bounds, b.bounds)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
