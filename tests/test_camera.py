try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np


def project(scene, points):
    """
    Project world-space points through the scene's camera to normalized
    device coordinates, the opposite of `Scene.camera_rays`.

    Parameters
    ----------
    scene : trimesh.Scene
      Scene with a camera.
    points : (n, 3) float
      World-space points.

    Returns
    ----------
    ndc : (n, 3) float
      Coordinates which are in view if all within [-1, 1].
    """
    homogeneous = g.np.column_stack([points, g.np.ones(len(points))])
    clip = (
        homogeneous
        @ g.np.linalg.inv(scene.camera_transform).T
        @ scene.camera.projection.T
    )
    return clip[:, :3] / clip[:, 3:4]


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

    def test_resolution_keeps_fov(self):
        """
        Setting a resolution keeps a set field of view, and only the view
        a scene is rendered with widens to the image with square pixels.
        """
        Camera = g.trimesh.scene.Camera
        current_view = g.trimesh.scene.cameras.current_view
        camera = Camera(resolution=[640, 480], fov=(60, 45), z_far=50.0)
        camera.resolution = (1920, 1080)
        copied = camera.copy()
        # a scene viewed at another shape keeps its camera as it was
        scene = g.trimesh.creation.box().scene()
        scene.camera = Camera(resolution=[640, 480], fov=(60, 45))
        scene.graph[scene.camera.name] = g.np.eye(4)
        viewed = current_view(scene, resolution=(1920, 1080))[0]
        half = g.np.tan(g.np.radians(viewed.fov) / 2)
        orthographic = g.trimesh.creation.box().scene()
        orthographic.camera = Camera(magnification=(2.0, 1.5), resolution=[640, 480])
        orthographic.graph[orthographic.camera.name] = g.np.eye(4)
        flat = current_view(orthographic, resolution=(1920, 1080))[0]
        # `set_camera` keeps the fov it is passed however often it's called
        framed = g.trimesh.creation.box().scene()
        for resolution in (None, (1920, 1080)):
            framed.set_camera(fov=(50, 40), resolution=resolution)
        assert g.np.array_equal(camera.fov, (60, 45)), "resolution keeps fov"
        assert g.np.array_equal(Camera(fov=(60, 45)).resolution, (1800, 1350)), (
            "default resolution"
        )
        assert (copied.z_near, copied.z_far) == (camera.z_near, 50), (
            "copy keeps clip planes"
        )
        assert g.np.array_equal(copied.fov, camera.fov), "copy keeps fov"
        assert g.np.array_equal(viewed.resolution, (1920, 1080)), (
            "view has the image resolution"
        )
        assert g.np.isclose(viewed.fov[1], 45.0), "view keeps vertical fov"
        assert g.np.isclose(half[0] / half[1], 1920 / 1080), "view has square pixels"
        assert g.np.array_equal(scene.camera.fov, (60, 45)) and g.np.array_equal(
            scene.camera.resolution, (640, 480)
        ), "scene camera unchanged"
        assert g.np.allclose(flat.magnification, (1.5 * 1920 / 1080, 1.5)), (
            "orthographic view keeps height"
        )
        assert g.np.array_equal(framed.camera.fov, (50, 40)) and g.np.array_equal(
            framed.camera.resolution, (1920, 1080)
        ), "set_camera twice keeps fov"

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

        # a scaled rotation frames from the same place as the unit one
        # and keeps its scale, as a camera node scaled with its scene
        rotation = g.trimesh.transformations.random_rotation_matrix(seed=3)
        for scale in (0.025, 40.0):
            scaled = rotation @ g.np.diag([scale] * 3 + [1])
            unit = g.trimesh.scene.cameras.look_at(moved, fov, rotation=rotation)
            framed = g.trimesh.scene.cameras.look_at(moved, fov, rotation=scaled)
            assert g.np.allclose(framed[:3, 3], unit[:3, 3]), f"{scale} same translation"
            assert g.np.allclose(framed[:3, :3], scaled[:3, :3]), f"{scale} keeps scale"

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

        # rays from a camera looking down -Z from 5 at a box whose top
        # face is at Z of 1, through a node which may scale the camera
        Camera = g.trimesh.scene.Camera
        box = g.trimesh.creation.box(extents=[40.0, 40.0, 2.0])
        lenses = {
            "perspective": {"fov": (60.0, 45.0)},
            "orthographic": {"magnification": (1.5, 1.0)},
        }
        width, height = 40, 30
        for kind, lens in lenses.items():
            for scale in (1.0, 0.025):
                name = f"{kind} at scale {scale}"
                scene = g.trimesh.Scene(box)
                scene.camera = Camera(resolution=(width, height), **lens)
                pose = g.trimesh.transformations.translation_matrix([0, 0, 5])
                scene.graph[scene.camera.name] = pose @ g.np.diag([scale] * 3 + [1])
                origins, vectors, pixels = scene.camera_rays()

                # the same view from an unscaled camera, which for an
                # orthographic camera sees `scale` times as much
                unscaled = g.trimesh.Scene(box)
                if kind == "orthographic":
                    lens = {"magnification": g.np.multiply(lens["magnification"], scale)}
                unscaled.camera = Camera(resolution=(width, height), **lens)
                unscaled.graph[unscaled.camera.name] = pose
                reference = unscaled.camera_rays()

                # the center of each pixel, counted from the bottom left
                column = (2.0 * pixels[:, 0] + 1.0) / width - 1.0
                row = (2.0 * pixels[:, 1] + 1.0) / height - 1.0
                distance = g.np.full(len(pixels), 4.0)
                if kind == "perspective":
                    half = g.np.tan(g.np.radians(lens["fov"]) / 2.0)
                    distance *= g.np.sqrt(
                        (column * half[0]) ** 2 + (row * half[1]) ** 2 + 1.0
                    )
                hits = origins + vectors * distance[:, None]
                ndc = project(scene, hits)
                assert g.np.allclose(g.np.linalg.norm(vectors, axis=1), 1.0), (
                    f"{name} unit rays"
                )
                assert (
                    g.np.allclose(origins, reference[0])
                    and g.np.allclose(vectors, reference[1])
                    and g.np.array_equal(pixels, reference[2])
                ), f"{name} equals unscaled"
                assert g.np.allclose(hits[:, 2], 1.0), f"{name} distance to the face"
                assert g.np.allclose(ndc[:, :2], g.np.column_stack((column, row))), (
                    f"{name} lands on pixel centers"
                )

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
        ndc = project(scene, g.trimesh.bounds.corners(scene.bounds))
        assert (g.np.abs(ndc) <= 1.0 + 1e-6).all(), "corners in frustum"
        assert g.np.linalg.norm(ndc[:, :2].mean(axis=0)) < 0.25, "centered"


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
