try:
    from . import generic as g
except BaseException:
    import generic as g


class CreationTest(g.unittest.TestCase):
    def setUp(self):
        engines = []
        if g.trimesh.util.has_module("triangle"):
            engines.append("triangle")
        if g.trimesh.util.has_module("mapbox_earcut"):
            engines.append("earcut")
        self.engines = engines

    def test_box(self):
        box = g.trimesh.creation.box

        # should create a unit cube with origin centroid
        m = box()
        assert g.np.allclose(m.bounds, [[-0.5] * 3, [0.5] * 3])

        # check creation by passing extents
        extents = g.np.array([1.2, 1.9, 10.3])
        m = box(extents=extents)
        assert g.np.allclose(m.extents, extents)
        assert g.np.allclose(m.bounds, [-extents / 2.0, extents / 2.0])

        bounds = g.np.array([[10.0, 11.9, 1.3], [100, 121, 53.002]])
        m = box(bounds=bounds)
        assert g.np.allclose(m.bounds, bounds)

    def test_cone(self):
        c = g.trimesh.creation.cone(radius=0.5, height=1.0)
        assert c.is_volume
        assert c.body_count == 1
        assert g.np.allclose(c.extents, 1.0, atol=0.03)
        assert c.metadata["shape"] == "cone"

    def test_cylinder(self):
        # tolerance for cylinders
        atol = 0.03

        c = g.trimesh.creation.cylinder(radius=0.5, height=1.0)
        assert c.is_volume
        assert c.body_count == 1
        assert g.np.allclose(c.extents, 1.0, atol=atol)
        assert c.metadata["shape"] == "cylinder"

        # check the "use a segment" feature
        # passed height should be overridden
        radius = 0.75
        offset = 10.0
        # true bounds
        bounds = ([[0, -radius, offset - radius], [1, radius, offset + radius]],)
        # create with a height that gets overridden
        c = g.trimesh.creation.cylinder(
            radius=radius, height=200, segment=[[0, 0, offset], [1, 0, offset]]
        )
        assert c.is_volume
        assert c.body_count == 1
        # make sure segment has been applied correctly
        assert g.np.allclose(c.bounds, bounds, atol=atol)
        # try again with no height passed
        c = g.trimesh.creation.cylinder(
            radius=radius, segment=[[0, 0, offset], [1, 0, offset]]
        )
        assert c.is_volume
        assert c.body_count == 1
        # make sure segment has been applied correctly
        assert g.np.allclose(c.bounds, bounds, atol=atol)

    def test_soup(self):
        count = 100
        mesh = g.trimesh.creation.random_soup(face_count=count)
        assert len(mesh.faces) == count
        assert len(mesh.face_adjacency) == 0
        assert len(mesh.split(only_watertight=True)) == 0
        assert len(mesh.split(only_watertight=False)) == count

    def test_capsule(self):
        mesh = g.trimesh.creation.capsule(radius=1.0, height=2.0)
        assert mesh.is_volume
        assert mesh.body_count == 1
        assert g.np.allclose(mesh.extents, [2, 2, 4], atol=0.05)

    def test_spheres(self):
        # test generation of UV spheres and icospheres
        for sphere in [g.trimesh.creation.uv_sphere(), g.trimesh.creation.icosphere()]:
            assert sphere.is_volume
            assert sphere.is_convex
            assert sphere.is_watertight
            assert sphere.is_winding_consistent

            assert sphere.body_count == 1
            assert sphere.metadata["shape"] == "sphere"

            # all vertices should have radius of exactly 1.0
            radii = g.np.linalg.norm(sphere.vertices - sphere.center_mass, axis=1)
            assert g.np.allclose(radii, 1.0)

        # test additional arguments
        red_sphere = g.trimesh.creation.icosphere(face_colors=(1.0, 0, 0))
        expected = g.np.full((len(red_sphere.faces), 4), (255, 0, 0, 255))
        g.np.testing.assert_allclose(red_sphere.visual.face_colors, expected)

    def test_camera_marker(self):
        """
        Create a marker including FOV for a camera object
        """
        # camera transform (pose) is identity
        camera = g.trimesh.scene.Camera(resolution=(320, 240), fov=(60, 45))
        meshes = g.trimesh.creation.camera_marker(camera=camera, marker_height=0.04)
        assert isinstance(meshes, list)
        # all meshes should be viewable type
        for mesh in meshes:
            assert isinstance(mesh, (g.trimesh.Trimesh, g.trimesh.path.Path3D))

    def test_axis(self):
        # specify the size of the origin radius
        origin_size = 0.04
        # specify the length of the cylinders
        axis_length = 0.4

        # construct a visual axis
        axis = g.trimesh.creation.axis(origin_size=origin_size, axis_length=axis_length)

        # AABB should be origin radius + cylinder length
        assert g.np.allclose(
            origin_size + axis_length, axis.bounding_box.primitive.extents, rtol=0.01
        )

    def test_path_sweep(self):
        if len(self.engines) == 0:
            return

        # Create base polygon
        vec = g.np.array([0, 1]) * 0.2
        n_comps = 100
        angle = g.np.pi * 2.0 / n_comps
        rotmat = g.np.array(
            [[g.np.cos(angle), -g.np.sin(angle)], [g.np.sin(angle), g.np.cos(angle)]]
        )
        perim = []
        for _i in range(n_comps):
            perim.append(vec)
            vec = g.np.dot(rotmat, vec)
        poly = g.Polygon(perim)

        # Create 3D path
        angles = g.np.linspace(0, 8 * g.np.pi, 1000)
        x = angles / 10.0
        y = g.np.cos(angles)
        z = g.np.sin(angles)
        path = g.np.c_[x, y, z]

        # Extrude
        for engine in self.engines:
            mesh = g.trimesh.creation.sweep_polygon(poly, path, engine=engine)
            assert mesh.is_volume

    def test_annulus(self):
        """
        Basic tests of annular cylinder creation
        """

        # run through transforms
        transforms = [None]
        transforms.extend(g.transforms)
        for T in transforms:
            a = g.trimesh.creation.annulus(r_min=1.0, r_max=2.0, height=1.0, transform=T)
            # mesh should be well constructed
            assert a.is_volume
            assert a.is_watertight
            assert a.is_winding_consistent
            assert a.metadata["shape"] == "annulus"

            # should be centered at origin
            assert g.np.allclose(a.center_mass, 0.0)
            # should be along Z
            axis = g.np.eye(3)
            if T is not None:
                # rotate the symmetry axis ground truth
                axis = g.trimesh.transform_points(axis, T)

            # should be along rotated Z
            assert g.np.allclose(a.symmetry_axis, axis[2]) or g.np.allclose(
                a.symmetry_axis, -axis[2]
            )

            radii = [g.np.dot(a.vertices, i) for i in axis[:2]]
            radii = g.np.linalg.norm(radii, axis=0)
            # vertices should all be at r_min or r_max
            assert g.np.logical_or(
                g.np.isclose(radii, 1.0), g.np.isclose(radii, 2.0)
            ).all()
            # all heights should be at +/- height/2.0
            assert g.np.allclose(g.np.abs(g.np.dot(a.vertices, axis[2])), 0.5)

        # do some cylinder comparison checks
        a = g.trimesh.creation.annulus(r_min=0.0, r_max=1.0, height=1.0)
        cylinder = g.trimesh.creation.cylinder(radius=1, height=1)
        # should survive a zero-inner-radius
        assert g.np.isclose(a.volume, cylinder.volume)
        assert g.np.isclose(a.area, cylinder.area)

        # bounds should be the same as a cylinder
        a = g.trimesh.creation.annulus(r_min=0.25, r_max=1.0, height=1.0)
        c = g.trimesh.creation.cylinder(radius=1, height=1)
        assert g.np.allclose(a.bounds, c.bounds)

        # segment should work the same for both
        seg = [[1, 2, 3], [4, 5, 6]]
        a = g.trimesh.creation.annulus(r_min=0.25, r_max=1.0, segment=seg)
        c = g.trimesh.creation.cylinder(radius=1, segment=seg)
        assert g.np.allclose(a.bounds, c.bounds)

    def test_triangulate(self):
        """
        Test triangulate using meshpy and triangle
        """
        # circles
        bigger = g.Point([10, 0]).buffer(1.0)
        smaller = g.Point([10, 0]).buffer(0.25)

        # circle with hole in center
        donut = bigger.difference(smaller)

        # make sure we have nonzero data
        assert bigger.area > 1.0
        # make sure difference did what we think it should
        assert g.np.isclose(donut.area, bigger.area - smaller.area)

        times = {"earcut": 0.0, "triangle": 0.0}
        iterations = 50
        # get a polygon to benchmark times with including interiors
        bench = [bigger, smaller, donut]
        bench.extend(g.get_mesh("2D/ChuteHolderPrint.DXF").polygons_full)
        bench.extend(g.get_mesh("2D/wrench.dxf").polygons_full)

        # check triangulation of both meshpy and triangle engine
        # including an example that has interiors
        for engine in self.engines:
            # make sure all our polygons triangulate resonably
            for poly in bench:
                v, f = g.trimesh.creation.triangulate_polygon(poly, engine=engine)
                # run asserts
                check_triangulation(v, f, poly.area)
                try:
                    # do a quick benchmark per engine
                    # in general triangle appears to be 2x
                    # faster than
                    times[engine] += (
                        min(
                            g.timeit.repeat(
                                "t(p, engine=e)",
                                repeat=3,
                                number=iterations,
                                globals={
                                    "t": g.trimesh.creation.triangulate_polygon,
                                    "p": poly,
                                    "e": engine,
                                },
                            )
                        )
                        / iterations
                    )
                except BaseException:
                    g.log.error("failed to benchmark triangle", exc_info=True)
        g.log.info(f"benchmarked triangulation on {len(bench)} polygons: {str(times)}")

    def test_triangulate_plumbing(self):
        """
        Check the plumbing of path triangulation
        """
        if len(self.engines) == 0:
            return
        p = g.get_mesh("2D/ChuteHolderPrint.DXF")
        for engine in self.engines:
            v, f = p.triangulate(engine=engine)
            check_triangulation(v, f, p.area)

    def test_truncated(self, count=10):
        # create some random triangles
        tri = g.random((count, 3, 3))

        m = g.trimesh.creation.truncated_prisms(tri)
        split = m.split()
        assert m.body_count == count
        assert len(split) == count
        assert all(s.volume > 0 for s in split)


def check_triangulation(v, f, true_area):
    assert g.trimesh.util.is_shape(v, (-1, 2))
    assert v.dtype.kind == "f"
    assert g.trimesh.util.is_shape(f, (-1, 3))
    assert f.dtype.kind == "i"

    tri = g.trimesh.util.stack_3D(v)[f]
    area = g.trimesh.triangles.area(tri).sum()
    assert g.np.isclose(area, true_area)


def test_torus():
    torus = g.trimesh.creation.torus

    major_radius = 1.0
    minor_radius = 0.2
    m = torus(major_radius=major_radius, minor_radius=minor_radius)

    extents = g.np.array(
        [
            2 * major_radius + 2 * minor_radius,
            2 * major_radius + 2 * minor_radius,
            2 * minor_radius,
        ]
    )
    assert g.np.allclose(m.extents, extents)
    assert g.np.allclose(m.bounds, [-extents / 2.0, extents / 2.0])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
