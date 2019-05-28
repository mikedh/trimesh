try:
    from . import generic as g
except BaseException:
    import generic as g

from io import BytesIO
from trimesh.exchange import binvox
from trimesh.voxel import runlength as rl
from trimesh import voxel as v


class BinvoxTest(g.unittest.TestCase):
    def test_load_save_invariance(self):
        np = g.np
        dense = np.random.uniform(size=(4, 4, 4)) > 0.8
        shape = dense.shape
        rl_data = rl.dense_to_rle(
            dense.transpose((0, 2, 1)).flatten(), dtype=np.uint8)
        translate = np.array([2, 5, 10], dtype=np.float32)
        scale = 3.6
        base = v.VoxelRle.from_binvox_data(rl_data, shape, translate, scale)
        np.testing.assert_equal(base.matrix.astype(np.bool), dense)
        file_obj = BytesIO(binvox.export_binvox(base))
        file_obj.seek(0)
        loaded = binvox.load_binvox(file_obj)
        np.testing.assert_equal(loaded.matrix.astype(np.bool), dense)
        self.assertTrue(isinstance(base, v.VoxelTranspose))
        self.assertTrue(isinstance(loaded, v.VoxelTranspose))
        self.assertTrue(isinstance(base.base, v.VoxelRle))
        self.assertTrue(isinstance(loaded.base, v.VoxelRle))
        np.testing.assert_equal(base.transpose_axes, loaded.transpose_axes)
        bb = base.base
        lb = loaded.base
        np.testing.assert_equal(bb.rle_data, lb.rle_data)
        np.testing.assert_equal(bb.shape, lb.shape)
        np.testing.assert_equal(bb.pitch, lb.pitch)
        np.testing.assert_equal(bb.origin, lb.origin)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
