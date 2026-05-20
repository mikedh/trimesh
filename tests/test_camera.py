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
            check = (np.ptp(points, axis=0)[:2] / 2.0) / g.np.tan(np.radians(fov / 2))
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

    def test_scaled_copy(self):
        s = g.get_mesh("cycloidal.3DXML")

        s.units = "mm"
        assert s.camera_transform.shape == (4, 4)

        # the camera node should have been removed on copy
        b = s.convert_units("m")
        assert b.camera_transform.shape == (4, 4)

    def test_default_framing(self):
        """
        Loading a complex assembly and asking for the default camera
        should leave the AABB completely inside the frustum and roughly
        centered. The pyglet2 viewer reads `scene.camera_transform` and
        `scene.camera.projection` each frame, so the chain that powers
        the renderer is exactly what we test here.
        """
        scene = g.get_mesh("cycloidal.3DXML")
        # accessing camera_transform implicitly calls set_camera() if
        # one hasn't been configured yet, so this drives the same default
        # path the viewer would.
        camera_transform = scene.camera_transform
        assert camera_transform.shape == (4, 4)

        # 8 AABB corners as world-space points.
        bounds = scene.bounds
        corners = g.np.array(
            [
                [x, y, z]
                for x in (bounds[0, 0], bounds[1, 0])
                for y in (bounds[0, 1], bounds[1, 1])
                for z in (bounds[0, 2], bounds[1, 2])
            ]
        )

        # `Scene.camera_project` is the helper the viewer uses internally; this
        # also documents the world -> view -> clip -> NDC chain.
        ndc = scene.camera_project(corners)
        assert ndc.shape == (8, 3)

        # every corner is inside the canonical frustum cube. a small
        # tolerance covers float round-trip wobble at the very edges.
        in_frame = (ndc >= -1.0 - 1e-6) & (ndc <= 1.0 + 1e-6)
        assert in_frame.all(), (
            f"AABB corners outside frustum:\n{ndc[~in_frame.all(axis=1)]}"
        )

        # roughly centered: the mean of all 8 NDC corners should be
        # close to the NDC origin in both x and y. depth (z) drifts
        # toward +1 because the AABB sits in front of the camera, so
        # we only constrain x and y.
        center_xy = ndc[:, :2].mean(axis=0)
        assert g.np.linalg.norm(center_xy) < 0.25, (
            f"AABB not centered in NDC: {center_xy}"
        )

        # cross-check: re-derive view + projection by hand and confirm
        # `scene.camera_project` produces an identical result. this is the
        # exact pair of matrices the pyglet2 viewer uploads every frame.
        view = g.np.linalg.inv(camera_transform)
        projection = scene.camera.projection
        homogeneous = g.np.column_stack([corners, g.np.ones(len(corners))])
        manual = homogeneous @ view.T @ projection.T
        manual_ndc = manual[:, :3] / manual[:, 3:4]
        assert g.np.allclose(ndc, manual_ndc, atol=1e-9)

    def test_save_image_matches_projection(self):
        """
        Render a simple convex mesh through `Scene.save_image` and
        confirm every projected vertex lands within one pixel of an
        actually-rendered (non-background) pixel. This locks down that
        the matrices the GPU sees match what `Scene.camera_project` returns.
        Exposes any bug in fov, aspect, or viewport conversion.

        Skipped unless `pyglet>=2` is importable and the platform has a
        usable EGL or X stack; pyglet 2 supports headless EGL on Linux
        so we ask for that explicitly.
        """
        try:
            import pyglet
        except ImportError:
            self.skipTest("pyglet not installed")
        if int(pyglet.version.split(".")[0]) < 2:
            self.skipTest("requires pyglet>=2")
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not installed")
        # ask pyglet for a hidden EGL context if no display is available
        pyglet.options["headless"] = True

        # use a simple, asymmetric, fully-convex mesh: an OBB has every
        # vertex on its silhouette, so we can demand 100% projection /
        # render agreement without worrying about back-face culling
        # eating interior vertices.
        import io

        mesh = g.trimesh.creation.box(extents=[3.0, 1.0, 2.0])
        scene = mesh.scene()

        try:
            png = scene.save_image(resolution=(800, 600))
        except BaseException as exc:
            self.skipTest(f"GL/EGL not usable in this environment: {exc}")
        if not png:
            self.skipTest("save_image returned empty bytes")

        image = g.np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        # background is white in the default viewer; anything < 250 in
        # any channel is rendered geometry.
        rendered = (image < 250).any(axis=-1)
        assert rendered.sum() > 0, "image came back empty"

        # project every vertex through the same camera the renderer
        # used. `save_image` may have updated `scene.camera.resolution`
        # to match the requested resolution.
        width, height = scene.camera.resolution
        ndc = scene.camera_project(mesh.vertices)
        in_frame = (g.np.abs(ndc) <= 1.0 + 1e-6).all(axis=1)
        assert in_frame.all(), "default camera should put a centered box in frustum"

        # NDC -> pixel: x left-to-right, y top-to-bottom (image origin
        # is top-left while NDC y points up).
        px = ((ndc[:, 0] + 1.0) * 0.5 * width).round().astype(int).clip(0, width - 1)
        py = ((1.0 - ndc[:, 1]) * 0.5 * height).round().astype(int).clip(0, height - 1)

        # 3x3 dilation: a vertex passes if any pixel within 1 of its
        # projected coordinate is rendered geometry.
        hits = g.np.zeros(len(mesh.vertices), dtype=bool)
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                neighbor_x = (px + delta_x).clip(0, width - 1)
                neighbor_y = (py + delta_y).clip(0, height - 1)
                hits |= rendered[neighbor_y, neighbor_x]
        assert hits.all(), (
            f"{(~hits).sum()}/{len(hits)} projected vertices missed the rendered geometry"
        )

    def test_save_image_fuze_centered(self):
        """
        Load `fuze.obj` with defaults and `Scene.save_image()` it. The
        rendered geometry must be:
          * fully contained in the framebuffer (every edge row/column
            empty, i.e. white pixels border the bottle on all sides),
          * agree with `Scene.camera_project()` (every projected vertex lands on
            or within a pixel of rendered geometry).

        This is the regression test for the HiDPI viewport bug where
        pyglet's `viewport` setter multiplied by `self.scale` and shoved
        the rendering off to the top-right of the framebuffer.
        """
        try:
            import pyglet
        except ImportError:
            self.skipTest("pyglet not installed")
        if int(pyglet.version.split(".")[0]) < 2:
            self.skipTest("requires pyglet>=2")
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not installed")
        pyglet.options["headless"] = True

        import io

        # default load + default render: exactly what `hi.py` does.
        scene = g.trimesh.load_scene(g.dir_models + "/fuze.obj")
        try:
            png = scene.save_image(resolution=(800, 600))
        except BaseException as exc:
            self.skipTest(f"GL/EGL not usable in this environment: {exc}")
        if not png:
            self.skipTest("save_image returned empty bytes")

        image = g.np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        rendered = (image < 250).any(axis=-1)
        assert rendered.sum() > 0, "image came back empty"

        # white pixels border the bottle on all sides: the four outer
        # edge rows/columns of the framebuffer must be background. if
        # the viewport is double-counted on HiDPI the geometry crashes
        # into a corner and at least one edge picks up rendered pixels.
        height, width = rendered.shape
        assert not rendered[0, :].any(), "top edge has rendered pixels"
        assert not rendered[height - 1, :].any(), "bottom edge has rendered pixels"
        assert not rendered[:, 0].any(), "left edge has rendered pixels"
        assert not rendered[:, width - 1].any(), "right edge has rendered pixels"

        # rendered bbox is roughly centered: the center of the bbox of
        # non-white pixels should sit near the framebuffer center.
        ys, xs = g.np.where(rendered)
        cx = (xs.min() + xs.max()) / 2.0
        cy = (ys.min() + ys.max()) / 2.0
        # within 5% of framebuffer center on each axis.
        assert abs(cx - width / 2.0) < 0.05 * width, (
            f"render not horizontally centered: bbox center x={cx} vs {width / 2}"
        )
        assert abs(cy - height / 2.0) < 0.05 * height, (
            f"render not vertically centered: bbox center y={cy} vs {height / 2}"
        )

        # every projected vertex lands within a pixel of rendered
        # geometry: same chain that `test_save_image_matches_projection`
        # exercises, but here the asset is a real textured OBJ rather
        # than a primitive box, so it also exercises the texture path.
        mesh = next(iter(scene.geometry.values()))
        fb_w, fb_h = scene.camera.resolution
        ndc = scene.camera_project(mesh.vertices)
        in_frame = (g.np.abs(ndc) <= 1.0 + 1e-6).all(axis=1)
        ndc = ndc[in_frame]
        px = ((ndc[:, 0] + 1.0) * 0.5 * fb_w).round().astype(int).clip(0, fb_w - 1)
        py = ((1.0 - ndc[:, 1]) * 0.5 * fb_h).round().astype(int).clip(0, fb_h - 1)
        hits = g.np.zeros(len(ndc), dtype=bool)
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                hits |= rendered[
                    (py + delta_y).clip(0, fb_h - 1),
                    (px + delta_x).clip(0, fb_w - 1),
                ]
        # the bottle has interior vertices (it's not convex) so demand a
        # very high majority rather than 100%.
        assert hits.mean() >= 0.95, (
            f"only {hits.mean():.1%} of projected fuze vertices hit rendered geometry"
        )

    def test_save_image_instanced_assembly(self):
        """
        Build a scene with three instances of one mesh + one instance
        of a second mesh (4 nodes, 2 unique geometries) and verify the
        per-instance world transforms are picked up by the renderer.

        For each scene-graph node, project its world-space vertices and
        confirm a healthy majority land on rendered pixels. With four
        distinct transforms covering both meshes this catches any bug
        where the instance index <-> matrix mapping has drifted.
        """
        try:
            import pyglet
        except ImportError:
            self.skipTest("pyglet not installed")
        if int(pyglet.version.split(".")[0]) < 2:
            self.skipTest("requires pyglet>=2")
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not installed")
        pyglet.options["headless"] = True

        import io

        # two unique meshes, multiple instances of each. icospheres
        # have every vertex on the silhouette, so a 1-pixel dilation
        # cleanly catches all of them. cubes have back-face vertices
        # that fall just outside the silhouette under perspective and
        # confuse the test without saying anything about instancing.
        small = g.trimesh.creation.icosphere(subdivisions=2, radius=0.6)
        large = g.trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        scene = g.trimesh.Scene()
        instance_transforms = []
        for index, offset in enumerate([(-2.5, 0, 0), (2.5, 0, 0), (0, 0, 2.5)]):
            transform = g.np.eye(4)
            transform[:3, 3] = offset
            instance_transforms.append((small, transform, f"small_{index}"))
            scene.add_geometry(
                small,
                transform=transform,
                geom_name="small",
                node_name=f"small_{index}",
            )
        large_transform = g.np.eye(4)
        large_transform[:3, 3] = [0, 2.5, 0]
        scene.add_geometry(
            large,
            transform=large_transform,
            geom_name="large",
            node_name="large_0",
        )
        instance_transforms.append((large, large_transform, "large_0"))

        try:
            png = scene.save_image(resolution=(800, 600))
        except BaseException as exc:
            self.skipTest(f"GL/EGL not usable: {exc}")
        if not png:
            self.skipTest("save_image returned empty bytes")

        image = g.np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        rendered = (image < 250).any(axis=-1)
        # require something visible per node
        assert rendered.sum() > 0, "render came back empty"

        width, height = scene.camera.resolution

        # for each scene-graph node, project its world-space vertices
        # and assert >=95% of in-frustum vertices land within 1 px of
        # rendered geometry. instances of the same mesh must each land
        # at their own transform-defined location, so this catches a
        # broken instance-index <-> matrix mapping.
        for mesh, transform, node_name in instance_transforms:
            world_vertices = g.trimesh.transformations.transform_points(
                mesh.vertices, transform
            )
            ndc = scene.camera_project(world_vertices)
            in_frame = (g.np.abs(ndc) <= 1.0 + 1e-6).all(axis=1)
            ndc = ndc[in_frame]
            if len(ndc) == 0:
                continue
            px = ((ndc[:, 0] + 1.0) * 0.5 * width).round().astype(int).clip(0, width - 1)
            py = (
                ((1.0 - ndc[:, 1]) * 0.5 * height).round().astype(int).clip(0, height - 1)
            )
            hits = g.np.zeros(len(ndc), dtype=bool)
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    neighbor_x = (px + delta_x).clip(0, width - 1)
                    neighbor_y = (py + delta_y).clip(0, height - 1)
                    hits |= rendered[neighbor_y, neighbor_x]
            hit_rate = hits.mean()
            assert hit_rate >= 0.95, (
                f"{node_name}: {hit_rate:.1%} of vertices on rendered "
                + "geometry (expected >=95%)"
            )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
