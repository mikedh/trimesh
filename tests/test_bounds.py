import generic as g

class BoundsTest(g.unittest.TestCase):
    def setUp(self):
        self.meshes = [g.trimesh.load(g.os.path.join(g.dir_models, 
                                          'featuretype.STL'))]
    def test_obb_mesh(self):
        '''
        Test the OBB functionality in attributes of Trimesh objects
        '''
        for m in self.meshes:
            for i in range(100):
                mat = g.trimesh.transformations.random_rotation_matrix()
                mat[0:3,3] = (g.np.random.random(3) -.5)* 100
                m.apply_transform(mat)

                box_ext = m.bounding_box_oriented.box_extents.copy()
                box_t = m.bounding_box_oriented.box_transform.copy()

                m.apply_transform(g.np.linalg.inv(box_t))

                test = m.bounds / (box_ext / 2.0)
                test_ok = g.np.allclose(test, [[-1,-1,-1],[1,1,1]])
                if not test_ok:
                    g.log.error('bounds test failed %s',
                                str(test))
                self.assertTrue(test_ok)
    def test_obb_points(self):
        '''
        Test OBB functions with raw points as input
        '''
        for dimension in [3,2]:
            for i in range(100):
                points = g.np.random.random((10,dimension))
                to_origin, extents = g.trimesh.bounds.oriented_bounds(points)

                self.assertTrue(g.trimesh.util.is_shape(to_origin, (dimension + 1, dimension + 1)))
                self.assertTrue(g.trimesh.util.is_shape(extents,   (dimension,)))

                transformed = g.trimesh.transform_points(points, to_origin)
                
                transformed_bounds = [transformed.min(axis=0),
                                      transformed.max(axis=0)]

                for i in transformed_bounds:
                    # assert that the points once our obb to_origin transform is applied
                    # has a bounding box centered on the origin
                    self.assertTrue(g.np.allclose(g.np.abs(i), extents/2.0))

                extents_tf = g.np.diff(transformed_bounds, axis=0).reshape(dimension)
                self.assertTrue(g.np.allclose(extents_tf,
                                              extents))
                

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
