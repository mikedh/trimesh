import generic as g


class VoxelTest(g.unittest.TestCase):

    def test_voxel(self):
        '''
        Test that voxels work at all
        '''
        for m in [g.get_mesh('featuretype.STL'),
                  g.trimesh.primitives.Box(),
                  g.trimesh.primitives.Sphere()]:
            for pitch in [.1, .1 - g.tol.merge]:
                v = m.voxelized(pitch)

                self.assertTrue(len(v.raw.shape) == 3)
                self.assertTrue(v.shape == v.raw.shape)
                self.assertTrue(v.volume > 0.0)

                self.assertTrue(v.origin.shape == (3,))
                self.assertTrue(isinstance(v.pitch, float))
                self.assertTrue(g.np.isfinite(v.pitch))

                self.assertTrue(g.np.issubdtype(v.filled_count, int))
                self.assertTrue(v.filled_count > 0)

                self.assertTrue(isinstance(v.mesh, g.trimesh.Trimesh))
                self.assertTrue(abs(v.mesh.volume - v.volume) < g.tol.merge)

                self.assertTrue(g.trimesh.util.is_shape(v.points, (-1, 3)))

                for p in v.points:
                    self.assertTrue(v.is_filled(p))

                outside = m.bounds[1] + m.scale
                self.assertFalse(v.is_filled(outside))

            g.log.info('Mesh volume was %f, voxelized volume was %f',
                       m.volume,
                       v.volume)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
