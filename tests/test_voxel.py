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

                assert len(v.matrix.shape) == 3
                assert v.shape == v.matrix.shape
                assert v.volume > 0.0

                assert v.origin.shape == (3,)
                assert isinstance(v.pitch, float)
                assert g.np.isfinite(v.pitch)

                assert isinstance(v.filled_count, int)
                assert v.filled_count > 0

                assert isinstance(v.as_boxes, g.trimesh.Trimesh)
                assert abs(v.as_boxes.volume - v.volume) < g.tol.merge

                assert g.trimesh.util.is_shape(v.points, (-1, 3))

                for p in v.points:
                    assert v.is_filled(p)

                outside = m.bounds[1] + m.scale
                assert not v.is_filled(outside)

                try:
                    cubes = v.marching_cubes
                    assert cubes.area > 0.0
                except ImportError:
                    g.log.info('no skimage, skipping marching cubes test')

            g.log.info('Mesh volume was %f, voxelized volume was %f',
                       m.volume,
                       v.volume)

    def test_marching(self):
        try:
            # make sure offset is correct
            matrix = g.np.ones((3, 3, 3), dtype=g.np.bool)
            mesh = g.trimesh.voxel.matrix_to_marching_cubes(
                matrix=matrix,
                pitch=1.0,
                origin=g.np.zeros(3))
            assert mesh.is_watertight

            mesh = g.trimesh.voxel.matrix_to_marching_cubes(
                matrix=matrix,
                pitch=3.0,
                origin=g.np.zeros(3))
            assert mesh.is_watertight

        except ImportError:
            g.log.info('no skimage, skipping marching cubes test')


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
