try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np


class CameraTests(g.unittest.TestCase):

    def test_K(self):
        resolution = (320, 240)
        fovxy = (60, 40)
        camera = g.trimesh.scene.Camera(
            resolution=resolution,
            fovxy=fovxy,
        )
        K_expected = np.array([
            [277.128,       0, 160],
            [      0, 329.697, 120],
            [      0,       0,   1],
        ], dtype=np.float64)
        np.testing.assert_allclose(camera.K, K_expected, rtol=1e-3)

    def test_consistency(self):
        resolution = (320, 240)
        fxfy = None
        fovxy = (60, 40)
        camera = g.trimesh.scene.Camera(
            resolution=resolution,
            fxfy=None,
            fovxy=fovxy,
        )
        camera = g.trimesh.scene.Camera(
            resolution=resolution,
            fxfy=camera.fxfy,
            fovxy=None
        )
        np.testing.assert_allclose(camera.fovxy, fovxy)
