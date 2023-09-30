try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.viewer.trackball import Trackball


class TrackballTest(g.unittest.TestCase):
    def test_resize(self):
        trackball = Trackball(
            pose=g.np.eye(4), size=(640, 480), scale=1.0, target=g.np.array([0, 0, 0])
        )
        trackball.down((320, 240))
        trackball.drag((321, 240))
        pose1 = trackball.pose
        trackball.drag((320, 240))
        trackball.resize((1280, 960))
        trackball.down((640, 480))
        trackball.drag((642, 480))
        pose2 = trackball.pose
        g.np.testing.assert_allclose(pose1, pose2)

    def test_drag_rotate(self):
        trackball = Trackball(
            pose=g.np.eye(4), size=(640, 480), scale=1.0, target=g.np.array([0, 0, 0])
        )

        # rotates around y-axis
        trackball.set_state(Trackball.STATE_ROTATE)
        trackball.down((320, 240))
        trackball.drag((321, 240))
        assert trackball.pose[0, 0] < 1
        assert trackball.pose[0, 1] == 0
        assert trackball.pose[0, 2] < 0
        assert trackball.pose[1, 0] == 0
        assert trackball.pose[1, 1] == 1
        assert trackball.pose[1, 2] == 0
        assert trackball.pose[2, 0] > 0
        assert trackball.pose[2, 1] == 0
        assert trackball.pose[2, 2] < 1
        g.np.testing.assert_allclose(trackball.pose[:, 3], [0, 0, 0, 1])
        g.np.testing.assert_allclose(trackball.pose[3, :], [0, 0, 0, 1])
        # rotate back
        trackball.drag((320, 240))
        g.np.testing.assert_allclose(trackball.pose, g.np.eye(4))

        # rotate around x-axis
        trackball.drag((320, 241))
        assert trackball.pose[0, 0] == 1
        assert trackball.pose[0, 1] == 0
        assert trackball.pose[0, 2] == 0
        assert trackball.pose[1, 0] == 0
        assert trackball.pose[1, 1] < 1
        assert trackball.pose[1, 2] < 0
        assert trackball.pose[2, 0] == 0
        assert trackball.pose[2, 1] > 0
        assert trackball.pose[2, 2] < 1
        g.np.testing.assert_allclose(trackball.pose[:, 3], [0, 0, 0, 1])
        g.np.testing.assert_allclose(trackball.pose[3, :], [0, 0, 0, 1])
        # rotate back
        trackball.drag((320, 240))
        g.np.testing.assert_allclose(trackball.pose, g.np.eye(4))

    def test_drag_roll(self):
        trackball = Trackball(
            pose=g.np.eye(4), size=(640, 480), scale=1.0, target=g.np.array([0, 0, 0])
        )

        # rotates around z-axis
        trackball.set_state(Trackball.STATE_ROLL)
        trackball.down((321, 241))
        trackball.drag((320, 241))
        assert trackball.pose[0, 0] < 1
        assert trackball.pose[0, 1] > 0
        assert trackball.pose[0, 2] == 0
        assert trackball.pose[1, 0] < 0
        assert trackball.pose[1, 1] < 1
        assert trackball.pose[1, 2] == 0
        assert trackball.pose[2, 0] == 0
        assert trackball.pose[2, 1] == 0
        assert trackball.pose[2, 2] == 1
        g.np.testing.assert_allclose(trackball.pose[:, 3], [0, 0, 0, 1])
        g.np.testing.assert_allclose(trackball.pose[3, :], [0, 0, 0, 1])
        # rotate back
        trackball.drag((321, 241))
        g.np.testing.assert_allclose(trackball.pose, g.np.eye(4))

    def test_drag_pan(self):
        trackball = Trackball(
            pose=g.np.eye(4), size=(640, 480), scale=1.0, target=g.np.array([0, 0, 0])
        )

        # translate to x
        trackball.set_state(Trackball.STATE_PAN)
        trackball.down((321, 241))
        trackball.drag((320, 241))
        g.np.testing.assert_allclose(trackball.pose[:3, :3], g.np.eye(3))
        assert trackball.pose[0, 3] != 0
        assert trackball.pose[1, 3] == 0
        assert trackball.pose[2, 3] == 0

    def test_drag_zoom(self):
        pose = g.trimesh.transformations.translation_matrix([0, 0, 1])
        trackball = Trackball(
            pose=pose, size=(640, 480), scale=1.0, target=g.np.array([0, 0, 0])
        )

        # translate to x
        trackball.set_state(Trackball.STATE_ZOOM)
        trackball.down((320, 240))
        trackball.drag((320, 241))
        g.np.testing.assert_allclose(trackball.pose[:3, :3], g.np.eye(3))
        assert trackball.pose[3, 0] == 0
        assert trackball.pose[3, 1] == 0
        assert trackball.pose[3, 2] < 1
        g.np.testing.assert_allclose(trackball.pose[3, :], [0, 0, 0, 1])

    def test_scroll(self):
        pose = g.trimesh.transformations.translation_matrix([0, 0, 1])
        trackball = Trackball(
            pose=pose, size=(640, 480), scale=1.0, target=g.np.array([0, 0, 0])
        )
        g.np.testing.assert_allclose(trackball.pose[:3, :3], g.np.eye(3))
        g.np.testing.assert_allclose(trackball.pose[:3, 3], [0, 0, 1])
        trackball.scroll(1)
        g.np.testing.assert_allclose(trackball.pose[:3, :3], g.np.eye(3))
        g.np.testing.assert_allclose(trackball.pose[:3, 3], [0, 0, 0.9])

    def test_rotate(self):
        trackball = Trackball(
            pose=g.np.eye(4), size=(640, 480), scale=1.0, target=g.np.array([0, 0, 0])
        )
        # rotates around y-axis
        trackball.rotate(g.np.deg2rad(1))
        assert trackball.pose[0, 0] < 1
        assert trackball.pose[0, 1] == 0
        assert trackball.pose[0, 2] > 0
        assert trackball.pose[1, 0] == 0
        assert trackball.pose[1, 1] == 1
        assert trackball.pose[1, 2] == 0
        assert trackball.pose[2, 0] < 0
        assert trackball.pose[2, 1] == 0
        assert trackball.pose[2, 2] < 1
        g.np.testing.assert_allclose(trackball.pose[:, 3], [0, 0, 0, 1])
        g.np.testing.assert_allclose(trackball.pose[3, :], [0, 0, 0, 1])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
