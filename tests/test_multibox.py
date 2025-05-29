try:
    from . import generic as g
except BaseException:
    import generic as g


class MultiboxTest(g.unittest.TestCase):
    def test_multibox_internal_face_removal(self):
        np = g.np

        # Create a small 2 * 2 * 2 binary voxelgrid
        voxelgrid = np.ones((2, 2, 2), dtype=bool)
        indices = np.argwhere(voxelgrid)
        points = g.trimesh.voxel.ops.indices_to_points(indices)

        # With internal faces (remove_internal_faces=False)
        mesh_w_internal_faces = g.trimesh.voxel.ops.multibox(
            points, pitch=1.0, remove_internal_faces=False
        )

        # Without internal faces (remove_internal_faces=True)
        mesh_wo_internal_faces = g.trimesh.voxel.ops.multibox(
            points, pitch=1.0, remove_internal_faces=True
        )
        self.assertLess(
            len(mesh_wo_internal_faces.faces), len(mesh_w_internal_faces.faces)
        )

        # Check bounding boxes are equal
        g.np.testing.assert_allclose(
            mesh_w_internal_faces.bounds, mesh_wo_internal_faces.bounds
        )

        # A 2 * 2 * 2 voxelgrid constitutes 8 voxels
        # Each voxel has 6 sides, with 2 triangular faces per side, totaling 6 * 2 = 12 triangular faces
        # 8 voxels * 12 triangular faces = 96 triangluar faces (when all the internal faces kept)
        self.assertEqual(len(mesh_w_internal_faces.faces), 96)

        # A 2 * 2 * 2 block has 6 exposed sides (i.e., top, bottom, front, back, left, right)
        # Each exposed side has 4 sub-sides corresponding to the 4 voxels, with 2 triangular faces per sub-side, totaling 4 * 2 = 8 triangular faces
        # 6 exposed sides * 8 triangular face = 48 triangular faces (when all the internal faces removed)
        self.assertEqual(len(mesh_wo_internal_faces.faces), 48)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
