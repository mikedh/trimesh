import generic as g

class VoxelTest(g.unittest.TestCase):

    def test_voxel(self):
        '''
        Test that voxels work at all
        '''
        for m in g.np.append(g.get_mesh('featuretype.STL'), g.trimesh.primitives.Box()):
            v = m.voxelized(.1)
            raw = v.raw
            self.assertTrue(len(v.raw.shape) == 3)
            self.assertTrue(v.volume > 0.0)

            self.assertTrue(v.origin.shape == (3,))
            self.assertTrue(isinstance(v.pitch, float))
            self.assertTrue(g.np.isfinite(v.pitch))
            
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
