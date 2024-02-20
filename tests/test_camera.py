try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np


class CameraTests(g.unittest.TestCase):
    def test_K(self):
        resolution = (320, 240)
        fov = (60, 40)
        camera = g.trimesh.scene.Camera(resolution=resolution, fov=fov)

        # ground truth matrix
        K_expected = np.array(
            [[277.128, 0, 160], [0, 329.697, 120], [0, 0, 1]], dtype=np.float64
        )

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
        camera = g.trimesh.scene.Camera(resolution=resolution, focal=focal, fov=fov)
        assert np.allclose(camera.fov, fov)
        camera = g.trimesh.scene.Camera(
            resolution=resolution, focal=camera.focal, fov=None
        )
        assert np.allclose(camera.fov, fov)

    def test_focal_updates_on_resolution_change(self):
        """
        Test changing resolution with set fov updates focal.
        """
        base_res = (320, 240)
        updated_res = (640, 480)
        fov = (60, 40)

        # start with initial data
        base_cam = g.trimesh.scene.Camera(resolution=base_res, fov=fov)
        # update both focal length and resolution
        base_focal = base_cam.focal
        base_cam.resolution = updated_res

        assert not g.np.allclose(base_cam.focal, base_focal)

        # camera created with same arguments should
        # have the same values
        new_cam = g.trimesh.scene.Camera(resolution=updated_res, fov=fov)
        assert g.np.allclose(base_cam.focal, new_cam.focal)

    def test_fov_updates_on_resolution_change(self):
        """
        Test changing resolution with set focal updates fov.
        """
        base_res = (320, 240)
        updated_res = (640, 480)
        focal = (100, 100)
        base_cam = g.trimesh.scene.Camera(resolution=base_res, focal=focal)
        base_fov = base_cam.fov
        base_cam.resolution = updated_res
        assert base_cam.fov is not base_fov
        new_cam = g.trimesh.scene.Camera(
            resolution=updated_res,
            focal=focal,
        )
        np.testing.assert_allclose(base_cam.fov, new_cam.fov)

    def test_lookat(self):
        """
        Test the "look at points" function
        """
        # original points
        ori = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

        for _i in range(10):
            # set the extents to be random but positive
            extents = g.random() * 10
            points = g.trimesh.util.stack_3D(ori.copy() * extents)

            fov = g.np.array([20, 50])

            # offset the points by a random amount
            offset = (g.random(3) - 0.5) * 100
            T = g.trimesh.scene.cameras.look_at(points + offset, fov)

            # check using trig
            check = (points.ptp(axis=0)[:2] / 2.0) / g.np.tan(np.radians(fov / 2))
            check += points[:, 2].mean()

            # Z should be the same as maximum trig option
            assert np.linalg.inv(T)[2, 3] >= check.max()

        # just run to test other arguments
        # TODO(unknown): find the way to test it correctly
        g.trimesh.scene.cameras.look_at(points, fov, center=points[0])
        g.trimesh.scene.cameras.look_at(points, fov, distance=1)

    def test_ray_index(self):
        # make sure to_rays is giving valid indexes
        s = g.trimesh.scene.Scene()
        res = g.np.array([512, 512])
        for i in range(0, 1000, 79):
            current = res + i
            s.camera.resolution = current
            # get ray index of camera
            rid = s.camera.to_rays()[1]
            assert all(rid.min(axis=0) == 0)
            assert all(rid.max(axis=0) == current - 1)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
