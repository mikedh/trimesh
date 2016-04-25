import generic as g

class BoundsTest(g.unittest.TestCase):
    def setUp(self):
        self.meshes = [g.trimesh.load(g.os.path.join(g.dir_models, 
                                          'featuretype.STL'))]
    def test_obb(self):
        for m in self.meshes:
            for i in range(100):
                mat = g.trimesh.transformations.random_rotation_matrix()
                mat[0:3,3] = (g.np.random.random(3) -.5)* 100
                m.transform(mat)

                box_ext = m.bounding_box_oriented.box_extents.copy()
                box_t = m.bounding_box_oriented.box_transform.copy()

                m.transform(g.np.linalg.inv(box_t))

                test = m.bounds / (box_ext / 2.0)
                self.assertTrue(g.np.allclose(test, [[-1,-1,-1],[1,1,1]]))

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
