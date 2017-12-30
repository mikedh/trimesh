import generic as g


class BoundsTest(g.unittest.TestCase):

    def setUp(self):
        meshes = [g.get_mesh(i) for i in ['large_block.STL',
                                          'featuretype.STL']]
        self.meshes = g.np.append(meshes, g.get_meshes(5))

    def test_obb_mesh(self):
        '''
        Test the OBB functionality in attributes of Trimesh objects
        '''
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
                self.assertTrue(test_ok)

                m.apply_transform(matrix)
                m.apply_obb()

                # after applying the obb, the extents of the AABB
                # should be the same as the OBB
                close = g.np.allclose(m.bounding_box.extents,
                                      m.bounding_box_oriented.extents,
                                      rtol=1e-3,
                                      atol=1e-3)
                if not close:
                    m.visual.face_colors = [200,0,0,100]
                    (m + m.bounding_box_oriented).show()
                    from IPython import embed
                    embed()
                    raise ValueError('OBB extents incorrect:\n{}\n{}'.format(
                        str(m.bounding_box.extents),
                        str(m.bounding_box_oriented.extents)))
                
            c = m.bounding_cylinder
            s = m.bounding_sphere
            p = m.bounding_primitive

    def test_obb_points(self):
        '''
        Test OBB functions with raw points as input
        '''
        for dimension in [3, 2]:
            for i in range(25):
                points = g.np.random.random((10, dimension))
                to_origin, extents = g.trimesh.bounds.oriented_bounds(points)

                self.assertTrue(g.trimesh.util.is_shape(
                    to_origin, (dimension + 1, dimension + 1)))
                self.assertTrue(g.trimesh.util.is_shape(
                    extents,   (dimension,)))

                transformed = g.trimesh.transform_points(points, to_origin)

                transformed_bounds = [transformed.min(axis=0),
                                      transformed.max(axis=0)]

                for i in transformed_bounds:
                    # assert that the points once our obb to_origin transform is applied
                    # has a bounding box centered on the origin
                    self.assertTrue(g.np.allclose(g.np.abs(i), extents / 2.0))

                extents_tf = g.np.diff(
                    transformed_bounds, axis=0).reshape(dimension)
                self.assertTrue(g.np.allclose(extents_tf,
                                              extents))


    def test_cylinder(self):
        '''
        '''
        height = 10.0
        radius = 1.0
        sphere = g.trimesh.util.grid_linspace([[0,0],
                                               [g.np.pi*2, g.np.pi*2]],
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


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
