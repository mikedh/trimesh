try:
    from . import generic as g
except BaseException:
    import generic as g


class VoxelGridTest(g.unittest.TestCase):
    def test_voxel(self):
        """
        Test that voxels work at all
        """
        for m in [
            g.get_mesh("featuretype.STL"),
            g.trimesh.primitives.Box(),
            g.trimesh.primitives.Sphere(),
        ]:
            for pitch in [0.1, 0.1 - g.tol.merge]:
                surface = m.voxelized(pitch=pitch)

                # make sure the voxelized pitch is similar to passed
                assert g.np.allclose(surface.pitch, pitch)

                for fill_method in ("base", "orthographic"):
                    solid = surface.copy().fill(method=fill_method)

                    assert len(surface.encoding.dense.shape) == 3
                    assert surface.shape == surface.encoding.dense.shape
                    assert surface.volume > 0.0

                    assert isinstance(surface.filled_count, int)
                    assert surface.filled_count > 0

                    box_surface = surface.as_boxes()
                    box_solid = solid.as_boxes()

                    assert isinstance(box_surface, g.trimesh.Trimesh)
                    assert abs(box_solid.volume - solid.volume) < g.tol.merge

                    assert g.trimesh.util.is_shape(surface.sparse_indices, (-1, 3))
                    assert len(solid.sparse_indices) >= len(surface.sparse_indices)
                    assert solid.sparse_indices.shape == solid.points.shape
                    outside = m.bounds[1] + m.scale
                    for vox in surface, solid:
                        assert vox.sparse_indices.shape == vox.points.shape
                        assert g.np.all(vox.is_filled(vox.points))
                        assert not vox.is_filled(outside)

                    try:
                        cubes = surface.marching_cubes
                        assert cubes.area > 0.0
                    except ImportError:
                        g.log.info("no skimage, skipping marching cubes test")

                    g.log.info(
                        "Mesh volume was %f, voxelized volume was %f",
                        m.volume,
                        surface.volume,
                    )

    def test_marching(self):
        """
        Test marching cubes on a matrix
        """
        try:
            from skimage import measure  # NOQA
        except ImportError:
            g.log.warning("no skimage, skipping marching cubes test")
            return

        # make sure offset is correct
        matrix = g.np.ones((3, 3, 3), dtype=bool)
        mesh = g.trimesh.voxel.ops.matrix_to_marching_cubes(matrix=matrix)
        assert mesh.is_watertight

        mesh = g.trimesh.voxel.ops.matrix_to_marching_cubes(matrix=matrix).apply_scale(
            3.0
        )
        assert mesh.is_watertight

    def test_marching_points(self):
        """
        Try marching cubes on points
        """
        try:
            from skimage import measure  # NOQA
        except ImportError:
            g.log.warning("no skimage, skipping marching cubes test")
            return

        # get some points on the surface of an icosahedron
        points = g.trimesh.creation.icosahedron().sample(1000)
        # make the pitch proportional to scale
        pitch = points.ptp(axis=0).min() / 10
        # run marching cubes
        mesh = g.trimesh.voxel.ops.points_to_marching_cubes(points=points, pitch=pitch)

        # mesh should have faces
        assert len(mesh.faces) > 0
        # mesh should be roughly centered
        assert (mesh.bounds[0] < -0.5).all()
        assert (mesh.bounds[1] > 0.5).all()

    def test_local(self):
        """
        Try calling local voxel functions
        """
        from trimesh.voxel import creation

        mesh = g.trimesh.creation.box()

        # it should have some stuff
        voxel = creation.local_voxelize(
            mesh=mesh, point=[0.5, 0.5, 0.5], pitch=0.1, radius=5, fill=True
        )

        assert len(voxel.shape) == 3

        # try it when it definitely doesn't hit anything
        empty = creation.local_voxelize(
            mesh=mesh, point=[10, 10, 10], pitch=0.1, radius=5, fill=True
        )
        # shouldn't have hit anything
        assert empty is None

        # try it when it is in the center of a volume
        creation.local_voxelize(
            mesh=mesh, point=[0, 0, 0], pitch=0.1, radius=2, fill=True
        )

    def test_points_to_from_indices(self):
        # indices = (points - origin) / pitch
        points = [[0, 0, 0], [0.04, 0.55, 0.39]]
        origin = [0, 0, 0]
        pitch = 0.1
        indices = [[0, 0, 0], [0, 6, 4]]

        # points -> indices
        indices2 = g.trimesh.voxel.ops.points_to_indices(
            points=points, origin=origin, pitch=pitch
        )

        g.np.testing.assert_allclose(indices, indices2, atol=0, rtol=0)

        # indices -> points
        points2 = g.trimesh.voxel.ops.indices_to_points(
            indices=indices, origin=origin, pitch=pitch
        )
        g.np.testing.assert_allclose(
            g.np.array(indices) * pitch + origin, points2, atol=0, rtol=0
        )
        g.np.testing.assert_allclose(points, points2, atol=pitch / 2 * 1.01, rtol=0)

        # indices -> points -> indices (this must be consistent)
        points2 = g.trimesh.voxel.ops.indices_to_points(
            indices=indices, origin=origin, pitch=pitch
        )
        indices2 = g.trimesh.voxel.ops.points_to_indices(
            points=points2, origin=origin, pitch=pitch
        )
        g.np.testing.assert_allclose(indices, indices2, atol=0, rtol=0)

    def test_as_boxes(self):
        voxel = g.trimesh.voxel

        pitch = 0.1
        origin = (0, 0, 1)

        matrix = g.np.eye(9, dtype=bool).reshape((-1, 3, 3))
        centers = g.trimesh.voxel.ops.matrix_to_points(
            matrix=matrix, pitch=pitch, origin=origin
        )
        v = voxel.VoxelGrid(matrix).apply_scale(pitch).apply_translation(origin)

        boxes1 = v.as_boxes()
        boxes2 = g.trimesh.voxel.ops.multibox(centers).apply_scale(pitch)
        colors = [g.trimesh.visual.DEFAULT_COLOR] * matrix.sum() * 12
        for boxes in [boxes1, boxes2]:
            g.np.testing.assert_allclose(boxes.visual.face_colors, colors, atol=0, rtol=0)

        # check assigning a single color
        color = [255, 0, 0, 255]
        boxes1 = v.as_boxes(colors=color)
        boxes2 = g.trimesh.voxel.ops.multibox(centers=centers, colors=color).apply_scale(
            pitch
        )
        colors = g.np.array([color] * len(centers) * 12)
        for boxes in [boxes1, boxes2]:
            g.np.testing.assert_allclose(boxes.visual.face_colors, colors, atol=0, rtol=0)

        # check matrix colors
        colors = color * g.np.ones(g.np.append(v.shape, 4), dtype=g.np.uint8)
        boxes = v.as_boxes(colors=colors)
        assert g.np.allclose(boxes.visual.face_colors, color, atol=0, rtol=0)

    def test_is_filled(self):
        """More rigorous test of VoxelGrid.is_filled."""
        n = 10
        matrix = g.np.random.uniform(size=(n + 1,) * 3) > 0.5
        not_matrix = g.np.logical_not(matrix)
        pitch = 1.0 / n
        origin = g.np.random.uniform(size=(3,))
        vox = g.trimesh.voxel.VoxelGrid(matrix)
        vox = vox.apply_scale(pitch).apply_translation(origin)
        not_vox = g.trimesh.voxel.VoxelGrid(not_matrix)
        not_vox = not_vox.apply_scale(pitch).apply_translation(origin)
        for a, b in ((vox, not_vox), (not_vox, vox)):
            points = a.points
            # slight jitter - shouldn't change indices
            points += (g.np.random.uniform(size=points.shape) - 1) * 0.4 * pitch
            g.np.random.shuffle(points)

            # all points are filled, and no empty points are filled
            assert g.np.all(a.is_filled(points))
            assert not g.np.any(b.is_filled(points))

            # test different number of dimensions
            points = g.np.stack([points, points[-1::-1]], axis=1)
            assert g.np.all(a.is_filled(points))
            assert not g.np.any(b.is_filled(points))

    def test_vox_sphere(self):
        # should be filled from 0-9
        matrix = g.np.ones((10, 10, 10))
        scale = 0.1
        vox = g.trimesh.voxel.VoxelGrid(matrix).apply_scale(scale)
        # epsilon from zero
        eps = 1e-4
        # should all be contained
        grid = g.trimesh.util.grid_linspace([[eps] * 3, [9 - eps] * 3], 11) * scale
        assert vox.is_filled(grid).all()

        # push it outside the filled area
        grid += 1.0
        assert not vox.is_filled(grid).any()

    def test_roundtrip(self):
        # try exporting and reloading in the "binvox" format
        m = g.trimesh.creation.box()
        v = m.voxelized(pitch=0.1)
        e = v.export(file_type="binvox")
        r = g.trimesh.load(file_obj=g.trimesh.util.wrap_as_stream(e), file_type="binvox")

        assert v.filled_count == r.filled_count
        assert g.np.allclose(r.bounds, v.bounds)

    def _test_equiv(self, v0, v1, query_points=None):
        """
        Test whether or not two `VoxelGrid` representation
        are consistent.

        Tests consistency of:
            shape
            filled_count
            volume
            matrix
            points
            origin
            pitch

        If query_points is provided, also tests
            is_filled
            points_to_indices
            indices_to_points

        Parameters
        ----------
        v0: `VoxelGrid` instance
        v1: `VoxelGrid` instance
        query_points: (optional) points as which `points_to_indices` and
        `is_filled` are tested for consistency.
        """

        def array_as_set(array2d):
            return {tuple(x) for x in array2d}

        # all points are filled
        assert g.np.all(v0.is_filled(v1.points))
        assert g.np.any(v1.is_filled(v0.points))

        # test different number of dimensions
        self.assertEqual(v0.shape, v1.shape)
        self.assertEqual(v0.filled_count, v1.filled_count)
        self.assertEqual(v0.volume, v1.volume)
        g.np.testing.assert_equal(v0.encoding.dense, v1.encoding.dense)
        # points will be in different order, but should contain same coords
        g.np.testing.assert_equal(array_as_set(v0.points), array_as_set(v1.points))
        # g.np.testing.assert_equal(v0.origin, v1.origin)
        # g.np.testing.assert_equal(v0.pitch, v1.pitch)
        if query_points is not None:
            indices0 = v0.points_to_indices(query_points)
            indices1 = v1.points_to_indices(query_points)
            g.np.testing.assert_equal(indices0, indices1)
            g.np.testing.assert_allclose(
                v0.points_to_indices(v0.indices_to_points(indices0)), indices0
            )
            g.np.testing.assert_allclose(
                v1.points_to_indices(v1.indices_to_points(indices1)), indices1
            )
            g.np.testing.assert_equal(
                v0.is_filled(query_points), v1.is_filled(query_points)
            )

    def test_voxel_rle(self):
        from trimesh.voxel import encoding as enc

        np = g.np
        voxel = g.trimesh.voxel
        shape = (4, 4, 4)
        rle_obj = enc.RunLengthEncoding(
            np.array([0, 8, 1, 40, 0, 16], dtype=np.uint8), dtype=bool
        )
        brle_obj = enc.BinaryRunLengthEncoding(np.array([8, 40, 16], dtype=np.uint8))
        v_rle = voxel.VoxelGrid(rle_obj.reshape(shape))
        self.assertEqual(v_rle.filled_count, 40)
        g.np.testing.assert_equal(
            v_rle.encoding.dense, g.np.reshape([0] * 8 + [1] * 40 + [0] * 16, shape)
        )

        v_brle = voxel.VoxelGrid(brle_obj.reshape(shape))
        query_points = g.np.random.uniform(size=(100, 3), high=4)
        self._test_equiv(v_rle, v_brle, query_points)

    def test_hollow(self):
        if not g.has_binvox:
            g.log.warning("no binvox to test!")
            return

        filled = g.trimesh.primitives.Sphere().voxelized(
            pitch=0.1, method="binvox", exact=True
        )
        hollow = filled.copy().hollow()
        self.assertLess(hollow.filled_count, filled.filled_count)
        self.assertGreater(hollow.filled_count, 0)

    def test_fill(self):
        from trimesh.voxel.morphology import fillers

        hollow = g.trimesh.primitives.Sphere().voxelized(pitch=0.1).hollow()

        for key in fillers:
            filled = hollow.copy().fill(key)
            self.assertLess(hollow.filled_count, filled.filled_count)

    def test_strip(self):
        if not g.has_binvox:
            g.log.warning("no binvox to test!")
            return

        octant = g.trimesh.primitives.Sphere().voxelized(
            pitch=0.1, method="binvox", exact=True
        )
        dense = octant.encoding.dense.copy()
        nx, ny, nz = octant.shape
        dense[: nx // 2] = 0
        dense[:, : ny // 2] = 0
        dense[:, :, nz // 2 :] = 0
        octant.encoding = dense
        stripped = octant.copy().strip()
        self.assertEqual(octant.filled_count, stripped.filled_count)
        self.assertEqual(octant.volume, stripped.volume)
        g.np.testing.assert_allclose(octant.points, stripped.points)
        self.assertGreater(octant.encoding.size, stripped.encoding.size)

    def test_binvox_with_dimension(self):
        if not g.has_binvox:
            g.log.warning("no binvox to test!")
            return

        dim = 10
        octant = g.trimesh.primitives.Sphere().voxelized(
            pitch=None, dimension=dim, method="binvox", exact=True
        )
        assert octant.shape == (dim, dim, dim)

    def test_transform_cache(self):
        encoding = [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 1], [0, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
        vg = g.trimesh.voxel.VoxelGrid(g.np.asarray(encoding))

        scale = g.np.asarray([12, 23, 24])
        s_matrix = g.np.eye(4)
        s_matrix[:3, :3] *= scale

        # original scale should be identity
        assert g.np.allclose(vg.scale, 1.0)

        # save the hash
        hash_ori = hash(vg._data)
        # modify the voxelgrid
        vg.apply_transform(s_matrix)

        # hash should have changed
        assert hash_ori != hash(vg._data)
        assert g.np.allclose(vg.scale, scale)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
