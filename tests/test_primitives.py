try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import triangle  # NOQA

    has_triangle = True
except ImportError:
    g.log.warning("Not testing extrude primitives!")
    has_triangle = False


class PrimitiveTest(g.unittest.TestCase):
    def setUp(self):
        self.primitives = []

        # do it with a flag in case there is more than one ImportError
        if has_triangle:
            e = g.trimesh.primitives.Extrusion()
            e.primitive.polygon = g.trimesh.path.polygons.random_polygon()
            e.primitive.height = 1.0
            self.primitives.append(e)

            self.primitives.append(
                g.trimesh.primitives.Extrusion(
                    polygon=g.trimesh.path.polygons.random_polygon(), height=293292.322
                )
            )

            self.primitives.append(e.buffer(0.25))

            self.primitives.append(
                g.trimesh.primitives.Extrusion(
                    polygon=g.Point([0, 0]).buffer(0.5), height=1.0
                )
            )

            self.primitives.append(
                g.trimesh.primitives.Extrusion(
                    polygon=g.Point([0, 0]).buffer(0.5), height=-1.0
                )
            )

        self.primitives.append(g.trimesh.primitives.Sphere())
        self.primitives.append(
            g.trimesh.primitives.Sphere(center=[0, 0, 100], radius=10.0, subdivisions=5)
        )
        self.primitives.append(g.trimesh.primitives.Box())

        try:
            self.primitives.append(
                g.trimesh.primitives.Box(
                    center=[102.20, 0, 102.0], extents=[29, 100, 1000]
                )
            )
            raise ValueError("Box shouldn't have accepted `center`!")
        except TypeError:
            # this should have raised a TypeError as `center` is not a kwarg
            pass

        self.primitives.append(
            g.trimesh.primitives.Box(
                extents=[10, 20, 30],
                transform=g.trimesh.transformations.random_rotation_matrix(),
            )
        )

        self.primitives.append(g.trimesh.primitives.Cylinder())
        self.primitives.append(
            g.trimesh.primitives.Cylinder(radius=10, height=1, sections=40)
        )

        self.primitives.append(g.trimesh.primitives.Capsule())
        self.primitives.append(g.trimesh.primitives.Capsule(radius=1.5, height=10))

    def test_scaling(self):
        # try a simple scaling test
        p = g.trimesh.primitives.Sphere(radius=1.0)
        m = p.to_mesh()
        assert g.np.allclose(p.extents, 2.0)
        assert g.np.allclose(m.extents, 2.0)
        p.apply_scale(0.5)
        m.apply_scale(0.5)
        assert g.np.isclose(p.primitive.radius, 0.5)
        assert g.np.allclose(p.extents, 1.0)
        assert g.np.allclose(m.extents, 1.0)
        assert g.np.allclose(p.extents, m.extents, atol=1e-3)

        p = g.trimesh.primitives.Box()
        p.apply_translation([0.5, 0.5, 0.5])
        p.apply_scale(5.0)
        assert g.np.allclose(p.bounds, [[0, 0, 0], [5, 5, 5]])

        try:
            raised = False
            p.apply_scale([2, 3, 4])
        except BaseException:
            raised = True
        if not raised:
            raise ValueError("primitives should raise on non-uniform scaling")

        # now try with more complicated generated data
        prims = [
            g.trimesh.primitives.Sphere(radius=1.0),
            g.trimesh.primitives.Sphere(radius=112.007),
            g.trimesh.primitives.Cylinder(radius=1.0, height=10.0),
            g.trimesh.primitives.Box(),
            g.trimesh.primitives.Box(extents=[12, 32, 31]),
            g.trimesh.primitives.Cylinder(radius=1.1212, height=0.001),
            g.trimesh.primitives.Capsule(radius=1.0, height=7.0),
        ]

        for original in prims:
            perm = [original, original.copy(), original.copy(), original.copy()]
            # try with a simple translation
            perm[1].primitive.transform = g.tf.translation_matrix([0, 0, 7])
            perm[2].apply_transform(g.tf.rotation_matrix(g.np.pi / 4, [0, 0, 1]))
            # try with a gnarly rotation
            perm[3].primitive.transform = g.tf.random_rotation_matrix(translate=1000)

            fields = set(dir(original.primitive))
            ori_radius, ori_height = None, None
            if "radius" in fields:
                ori_radius = original.primitive.radius
            if "height" in fields:
                ori_height = original.primitive.height

            for scale in [1e-2, 0.123, 0.5, 100.2]:
                for po in perm:
                    # converting to mesh will do all scaling
                    # and transformations on simple discrete
                    # copy of primitive and should match with
                    # only tessellation differences
                    p = po.copy()
                    m = p.to_mesh()

                    # make sure we have the types we think we do
                    assert isinstance(p, g.trimesh.primitives.Primitive)
                    assert isinstance(m, g.trimesh.Trimesh)

                    assert g.np.allclose(p.extents, m.extents)

                    p.apply_scale(scale)
                    m.apply_scale(scale)

                    # matrix should never have scale
                    assert g.tf.is_rigid(p.primitive.transform)

                    if ori_radius is not None:
                        assert g.np.isclose(p.primitive.radius, ori_radius * scale)
                    if ori_height is not None:
                        assert g.np.isclose(p.primitive.height, ori_height * scale)

                    # should be the same size
                    assert g.np.allclose(p.extents, m.extents, atol=1e-3 * scale)
                    # should be in the same place
                    assert g.np.allclose(p.bounds, m.bounds, atol=1e-3 * scale)

    def test_mesh_schema(self):
        # this schema should define every primitive.
        schema = g.trimesh.resources.get_schema("primitive/trimesh.schema.json")
        # make sure a mesh passes the schema
        m = g.trimesh.creation.box()
        g.jsonschema.validate(m.to_dict(), schema)

    def test_primitives(self):
        kind = {i.__class__.__name__ for i in self.primitives}
        # make sure our test data has every primitive
        kinds = {"Box", "Capsule", "Cylinder", "Sphere"}
        if has_triangle:
            kinds.add("Extrusion")
        assert kind == kinds

        # this schema should define every primitive.
        schema = g.trimesh.resources.get_schema("primitive/primitive.schema.json")

        for primitive in self.primitives:
            # convert to a dict
            d = primitive.to_dict()
            # validate the output of the to-dict method
            g.jsonschema.validate(d, schema)

            # just triple-check that we have a transform
            # this should have been validated by the schema
            assert g.np.shape(d["transform"]) == (4, 4)
            assert g.trimesh.transformations.is_rigid(d["transform"])
            # make sure the value actually json-dumps
            assert len(g.json.dumps(d)) > 0

            # make sure faces and vertices are correct
            assert g.trimesh.util.is_shape(primitive.faces, (-1, 3))
            assert g.trimesh.util.is_shape(primitive.vertices, (-1, 3))
            # check dtype of faces and vertices
            assert primitive.faces.dtype.kind == "i"
            assert primitive.vertices.dtype.kind == "f"

            assert primitive.volume > 0.0
            assert primitive.area > 0.0

            # convert to base class trimesh
            as_mesh = primitive.to_mesh()

            assert as_mesh.volume > 0.0
            assert as_mesh.area > 0.0

            assert g.np.allclose(primitive.extents, as_mesh.extents)
            assert g.np.allclose(primitive.bounds, as_mesh.bounds)

            assert g.np.isclose(primitive.volume, as_mesh.volume, rtol=0.05)
            assert g.np.isclose(primitive.area, as_mesh.area, rtol=0.05)

            assert primitive.is_winding_consistent
            assert primitive.is_watertight
            assert as_mesh.is_winding_consistent
            assert as_mesh.is_watertight

            # check that overload of dir worked
            assert len([i for i in dir(primitive.primitive) if "_" not in i]) > 0
            if hasattr(primitive, "direction"):
                assert primitive.direction.shape == (3,)

            centroid = primitive.centroid.copy()
            translation = [0, 0, 5]
            primitive.apply_translation(translation)

            # centroid should have translated correctly
            assert g.np.allclose(primitive.centroid - centroid, translation)

    def test_sample(self):
        transform = g.trimesh.transformations.random_rotation_matrix()
        box = g.trimesh.primitives.Box(transform=transform, extents=[20, 10, 100])
        for kwargs in [
            {"step": 8},
            {"step": [10, 0.4, 10]},
            {"count": 8},
            {"count": [10, 3, 5]},
        ]:
            grid = box.sample_grid(**kwargs)
            assert g.trimesh.util.is_shape(grid, (-1, 3))
            assert (box.nearest.signed_distance(grid) > -1e-6).all()

    def test_box(self):
        """
        Test the setter on box primitives
        """
        start = [20, 10, 100]
        box = g.trimesh.primitives.Box(extents=start)
        assert g.np.allclose(box.primitive.extents, start)
        assert g.np.allclose(box.extents, start)
        if g.has_path:
            # check to see if outline function works
            assert g.np.allclose(box.as_outline().extents, start)

    def test_transform_return(self):
        # make sure primitives return `self`
        Box = g.trimesh.primitives.Box

        assert isinstance(Box().apply_transform(g.np.eye(4)), Box)
        assert isinstance(Box().apply_transform(g.transforms[0]), Box)

    def test_cyl_buffer(self):
        # test our inflation of cylinder primitives
        c = g.trimesh.primitives.Cylinder(
            radius=1.0,
            height=10.0,
            transform=g.trimesh.transformations.random_rotation_matrix(),
        )
        # inflate cylinder
        b = c.buffer(1.0)
        assert g.np.isclose(b.primitive.height, 12.0)
        assert g.np.isclose(b.primitive.radius, 2.0)
        # should contain all vertices of source mesh
        assert b.contains(c.vertices).all()
        # should contain line segment
        assert b.contains(c.segment).all()

    def test_transform_attribute(self):
        for primitive in self.primitives:
            assert hasattr(primitive, "transform")

            assert g.trimesh.util.is_shape(primitive.transform, (4, 4))

            if hasattr(primitive.primitive, "center"):
                assert g.np.allclose(
                    primitive.primitive.center, primitive.transform[:3, 3]
                )

    def test_sphere_center(self):
        s = g.trimesh.primitives.Sphere(center=[0, 0, 100], radius=10.0, subdivisions=5)
        assert g.np.allclose(s.center, [0, 0, 100])

        s.center = [1, 1, 1]
        assert g.np.allclose(s.center, [1, 1, 1])

        s.center[:2] = [0, 3]
        assert g.np.allclose(s.center, [0, 3, 1])

    def test_box_bounds_constructor(self):
        # check to see that we can construct a box using AABB
        bounds = [[0.2, 0.3, 0.4], [10, 11, 12.2]]
        prim = g.trimesh.primitives.Box(bounds=bounds)

        # bounds should match requesta
        assert g.np.allclose(prim.bounds, bounds)

        try:
            # should raise a ValueError
            g.trimesh.primitives.Box(extents=bounds[0], bounds=bounds)
            raise AssertionError("box shouldn't have accepted both bounds and extents!")
        except ValueError:
            pass

    def test_copy(self):
        start = [20, 10, 100]
        # check that copy preserves overridden density and center-mass
        # for both Primitive objects and regular Trimesh objects.
        meshes = [
            g.trimesh.primitives.Box(extents=start),
            g.trimesh.creation.box(extents=start),
        ]

        for box in meshes:
            box.density = 0.3
            box.center_mass = g.np.array([0.1, -0.6, 11.3])
            box.metadata["foo"] = "bar"
            box_copy = box.copy()
            assert box.density == box_copy.density
            assert g.np.allclose(box.center_mass, box_copy.center_mass)
            assert box.metadata["foo"] == box_copy.metadata["foo"]


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
