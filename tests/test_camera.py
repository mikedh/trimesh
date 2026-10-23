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

    def test_resolution_keeps_pixels_square(self):
        """A new shape keeps the vertical fov and changes the horizontal one."""
        camera = g.trimesh.scene.Camera(resolution=[640, 480], fov=(60, 45), z_far=50.0)
        # the ratio of the half angle tangents is the shape of square pixels
        before = g.np.tan(g.np.radians(camera.fov) / 2)
        camera.resolution = (1920, 1080)
        after = g.np.tan(g.np.radians(camera.fov) / 2)
        copied = camera.copy()
        same = g.trimesh.scene.Camera(resolution=[320, 240], fov=(60, 45))
        checks = {
            "vertical fov kept": g.np.isclose(camera.fov[1], 45.0),
            "horizontal fov follows the shape": g.np.isclose(
                (after[0] / after[1]) / (before[0] / before[1]),
                (1920 / 1080) / (640 / 480),
            ),
            "same shape is unchanged": g.np.allclose(same.fov, (60, 45)),
            "copy keeps clip planes": (copied.z_near, copied.z_far)
            == (camera.z_near, 50),
            "copy keeps fov": g.np.allclose(copied.fov, camera.fov),
        }
        for name, ok in checks.items():
            assert ok, name

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

        # draw inside the block: `g.random` gave every iteration the same
        # extents and offset so this tested one camera ten times
        with g.RandomSeed() as r:
            for _i in range(10):
                # set the extents to be random but positive
                extents = r.random() * 10
                points = g.trimesh.util.stack_3D(ori.copy() * extents)

                fov = g.np.array([20, 50])

                # offset the points by a random amount
                offset = (r.random(3) - 0.5) * 100
                moved = points + offset
                T = g.trimesh.scene.cameras.look_at(moved, fov)

                # check using trig
                check = (np.ptp(points, axis=0)[:2] / 2.0) / g.np.tan(np.radians(fov / 2))

                # `look_at` returns a camera-to-world pose so the distance is
                # from the camera back to what it's pointed at: comparing the
                # world-to-camera translation against this only ever passed
                # because the offset was a fixed large negative number
                distance = T[2, 3] - moved[:, 2].mean()

                # camera should be far enough back to fit every point in frame
                assert distance + 1e-8 >= check.max()

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

    def test_scaled_copy(self):
        s = g.get_mesh("cycloidal.3DXML")

        s.units = "mm"
        assert s.camera_transform.shape == (4, 4)

        # the camera node should have been removed on copy
        b = s.convert_units("m")
        assert b.camera_transform.shape == (4, 4)

    def test_default_framing(self):
        """The default camera centers a whole assembly inside its frustum."""
        scene = g.get_mesh("cycloidal.3DXML")
        ndc = scene.camera_project(g.trimesh.bounds.corners(scene.bounds))
        checks = {
            "corners in frustum": (g.np.abs(ndc) <= 1.0 + 1e-6).all(),
            "centered": g.np.linalg.norm(ndc[:, :2].mean(axis=0)) < 0.25,
        }
        for name, ok in checks.items():
            assert ok, name

    def test_save_image(self):
        """`save_image` renders exactly what the camera sees and leaves it as it was."""
        g.require_rendering()
        translation = g.trimesh.transformations.translation_matrix
        sphere = g.trimesh.creation.icosphere
        # a small sphere instanced at three nodes and a large one
        assembly = g.trimesh.Scene({"small": sphere(2, 0.6), "large": sphere(2, 1.0)})
        offsets = {"small": (-2.5, 0, 0), "small_1": (2.5, 0, 0), "small_2": (0, 0, 2.5)}
        for node, offset in offsets.items():
            assembly.graph.update(
                frame_to=node, matrix=translation(offset), geometry="small"
            )
        assembly.graph.update(frame_to="large", matrix=translation([0, 2.5, 0]))
        scenes = {
            "box": g.trimesh.creation.box(extents=[3.0, 1.0, 2.0]).scene(),
            "fuze": g.trimesh.load_scene(g.dir_models + "/fuze.obj"),
            "assembly": assembly,
        }
        checks = {}
        for name, scene in scenes.items():
            # without the floor every drawn pixel is the scene's geometry
            g.set_render(scene, ground_shadow=False)
            before = [*scene.camera.resolution, *scene.camera.fov]
            image = g.png(scene.save_image(resolution=[400, 300]))
            checks[f"{name} size"] = image.shape[:2] == (300, 400)
            after = [*scene.camera.resolution, *scene.camera.fov]
            checks[f"{name} camera unchanged"] = g.np.allclose(after, before)
            # cast the reference rays through the camera the image used
            scene.camera.resolution = (400, 300)
            checks.update(g.silhouette_checks(scene, image, f"{name} "))
        for name, ok in checks.items():
            assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
