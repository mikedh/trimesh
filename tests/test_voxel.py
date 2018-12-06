try:
    from . import generic as g
except BaseException:
    import generic as g


class VoxelTest(g.unittest.TestCase):

    def test_voxel(self):
        """
        Test that voxels work at all
        """
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

                box = v.as_boxes(solid=False)
                boxF = v.as_boxes(solid=True)

                assert isinstance(box, g.trimesh.Trimesh)
                assert abs(boxF.volume - v.volume) < g.tol.merge

                assert g.trimesh.util.is_shape(v.points, (-1, 3))

                assert len(v.sparse_solid) > len(v.sparse_surface)

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

    def test_local(self):
        """
        Try calling local voxel functions
        """
        mesh = g.trimesh.creation.box()

        # it should have some stuff
        voxel = g.trimesh.voxel.local_voxelize(mesh=mesh,
                                               point=[.5, .5, .5],
                                               pitch=.1,
                                               radius=5,
                                               fill=True)

        assert len(voxel[0].shape) == 3

        # try it when it definitly doesn't hit anything
        empty = g.trimesh.voxel.local_voxelize(mesh=mesh,
                                               point=[10, 10, 10],
                                               pitch=.1,
                                               radius=5,
                                               fill=True)

        # try it when it is in the center of a volume
        center = g.trimesh.voxel.local_voxelize(mesh=mesh,
                                                point=[0, 0, 0],
                                                pitch=.1,
                                                radius=2,
                                                fill=True)

    def test_points_to_from_indices(self):
        points = [[0.05, 0.55, 0.35]]
        origin = [0, 0, 0]
        pitch = 0.1
        indices = [[1, 6, 4]]

        # points -> indices
        indices2 = g.trimesh.voxel.points_to_indices(
            points=points, origin=origin, pitch=pitch
        )
        g.np.testing.assert_allclose(indices, indices2, atol=0, rtol=0)

        # indices -> points
        points2 = g.trimesh.voxel.indices_to_points(
            indices=indices, origin=origin, pitch=pitch
        )
        g.np.testing.assert_allclose(points, points2)

        # indices -> points -> indices (this must be consistent)
        points2 = g.trimesh.voxel.indices_to_points(
            indices=indices, origin=origin, pitch=pitch
        )
        indices2 = g.trimesh.voxel.points_to_indices(
            points=points2, origin=origin, pitch=pitch
        )
        g.np.testing.assert_allclose(indices, indices2, atol=0, rtol=0)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
