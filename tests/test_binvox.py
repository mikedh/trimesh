try:
    from . import generic as g
except BaseException:
    import generic as g

from io import BytesIO

from trimesh import voxel as v
from trimesh.exchange import binvox
from trimesh.voxel import runlength as rl


class BinvoxTest(g.unittest.TestCase):
    def test_load_save_invariance(self):
        np = g.np
        n = 4
        dense = np.random.uniform(size=(n,) * 3) > 0.8
        dense[0, 0, 0] = dense[-1, -1, -1] = 1  # ensure extent test works
        shape = dense.shape
        rl_data = rl.dense_to_rle(dense.flatten(), dtype=np.uint8)
        translate = np.array([2, 5, 10], dtype=np.float32)
        scale = 5.0
        base = binvox.voxel_from_binvox(
            rl_data, shape, translate, scale, axis_order="xzy"
        )
        s = scale / (n - 1)
        np.testing.assert_equal(
            base.transform,
            np.array([[s, 0, 0, 2], [0, s, 0, 5], [0, 0, s, 10], [0, 0, 0, 1]]),
        )
        dense = dense.transpose((0, 2, 1))
        bound_min = translate - 0.5 * s
        bound_max = translate + scale + 0.5 * s
        np.testing.assert_allclose(base.bounds, [bound_min, bound_max])
        np.testing.assert_equal(base.encoding.dense, dense)

        if binvox.binvox_encoder is None:
            g.log.warning("No binvox encoder found, skipping binvox export tests")
            return

        file_obj = BytesIO(binvox.export_binvox(base))
        file_obj.seek(0)
        loaded = binvox.load_binvox(file_obj)
        np.testing.assert_equal(loaded.encoding.dense, base.encoding.dense)
        self.assertTrue(isinstance(base, v.VoxelGrid))
        self.assertTrue(isinstance(loaded, v.VoxelGrid))
        np.testing.assert_equal(base.transform, loaded.transform)
        np.testing.assert_equal(base.shape, loaded.shape)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
