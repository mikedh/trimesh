try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np


class CameraTests(g.unittest.TestCase):

    def test_K(self):
        resolution = (320, 240)
        fov = (60, 40)
        camera = g.trimesh.scene.Camera(
            resolution=resolution,
            fov=fov)

        # ground truth matrix
        K_expected = np.array([[277.128, 0, 160],
                               [0, 329.697, 120],
                               [0, 0, 1]],
                              dtype=np.float64)

        assert np.allclose(camera.K, K_expected, rtol=1e-3)

        # check to make sure assignment from matrix works
        K_set = K_expected.copy()
        K_set[:2, 2] = 300
        camera.K = K_set
        assert np.allclose(camera.resolution, 600)

    def test_consistency(self):
        resolution = (320, 240)
        focal = None
        fov = (60, 40)
        camera = g.trimesh.scene.Camera(
            resolution=resolution,
            focal=None,
            fov=fov)
        camera = g.trimesh.scene.Camera(
            resolution=resolution,
            focal=camera.focal,
            fov=None)
        assert np.allclose(camera.fov, fov)

    def test_lookat(self):
        """
        Test the "look at points" function
        """
        # original points
        ori = np.array([[-1, -1],
                        [1, -1],
                        [1, 1],
                        [-1, 1]])

        for i in range(10):
            # set the extents to be random but positive
            extents = g.random() * 10
            points = g.trimesh.util.stack_3D(ori.copy() * extents)

            fov = g.np.array([20, 50])

            # offset the points by a random amount
            offset = (g.random(3) - 0.5) * 100
            T = g.trimesh.scene.cameras.look_at(points + offset, fov)

            # check using trig
            check = (points.ptp(axis=0)[:2] / 2.0) / \
                g.np.tan(np.radians(fov / 2))
            check += points[:, 2].mean()

            # Z should be the same as maximum trig option
            assert g.np.isclose(T[2, 3], check.max())


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
