try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import triangle  # NOQA
    has_triangle = True
except ImportError:
    g.log.warning('No triangle! Not testing extrude primitives!')
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
                    polygon=g.trimesh.path.polygons.random_polygon(),
                    height=293292.322))

            self.primitives.append(e.buffer(.25))

            self.primitives.append(
                g.trimesh.primitives.Extrusion(
                    polygon=g.Point([0, 0]).buffer(.5),
                    height=1.0))

            self.primitives.append(
                g.trimesh.primitives.Extrusion(
                    polygon=g.Point([0, 0]).buffer(.5),
                    height=-1.0))

        self.primitives.append(g.trimesh.primitives.Sphere())
        self.primitives.append(g.trimesh.primitives.Sphere(center=[0, 0, 100],
                                                           radius=10.0,
                                                           subdivisions=5))
        self.primitives.append(g.trimesh.primitives.Box())
        self.primitives.append(
            g.trimesh.primitives.Box(
                center=[102.20, 0, 102.0],
                extents=[29, 100, 1000]))
        self.primitives.append(g.trimesh.primitives.Box(
            extents=[10, 20, 30],
            transform=g.trimesh.transformations.random_rotation_matrix()))

        self.primitives.append(g.trimesh.primitives.Cylinder())
        self.primitives.append(g.trimesh.primitives.Cylinder(radius=10,
                                                             height=1,
                                                             sections=40))

        self.primitives.append(g.trimesh.primitives.Capsule())
        self.primitives.append(g.trimesh.primitives.Capsule(radius=1.5,
                                                            height=10))

    def test_primitives(self):
        for primitive in self.primitives:
            # make sure faces and vertices are correct
            assert g.trimesh.util.is_shape(primitive.faces,
                                           (-1, 3))
            assert g.trimesh.util.is_shape(primitive.vertices,
                                           (-1, 3))
            # check dtype of faces and vertices
            assert primitive.faces.dtype.kind == 'i'
            assert primitive.vertices.dtype.kind == 'f'

            assert primitive.volume > 0.0
            assert primitive.area > 0.0

            # convert to base class trimesh
            as_mesh = primitive.to_mesh()

            try:
                assert as_mesh.volume > 0.0
                assert as_mesh.area > 0.0
            except BaseException:
                from IPython import embed
                embed()

            assert g.np.allclose(primitive.extents,
                                 as_mesh.extents)
            assert g.np.allclose(primitive.bounds,
                                 as_mesh.bounds)

            assert g.np.isclose(primitive.volume,
                                as_mesh.volume,
                                rtol=.05)
            assert g.np.isclose(primitive.area,
                                as_mesh.area,
                                rtol=.05)

            assert primitive.is_winding_consistent
            assert primitive.is_watertight
            assert as_mesh.is_winding_consistent
            assert as_mesh.is_watertight

            # check that overload of dir worked
            assert len([i
                        for i in dir(primitive.primitive) if '_' not in i]) > 0

            if hasattr(primitive, 'direction'):
                assert primitive.direction.shape == (3,)

            centroid = primitive.centroid.copy()
            translation = [0, 0, 5]
            primitive.apply_translation(translation)

            # centroid should have translated correctly
            assert g.np.allclose(primitive.centroid - centroid,
                                 translation)

    def test_sample(self):
        transform = g.trimesh.transformations.random_rotation_matrix()
        box = g.trimesh.primitives.Box(transform=transform,
                                       extents=[20, 10, 100])
        for kwargs in [{'step': 8},
                       {'step': [10, .4, 10]},
                       {'count': 8},
                       {'count': [10, 3, 5]}]:
            grid = box.sample_grid(**kwargs)
            assert g.trimesh.util.is_shape(grid, (-1, 3))
            assert (box.nearest.signed_distance(grid) > -1e-6).all()

    def test_box(self):
        """
        Test the setter on box primitives
        """
        start = [20, 10, 100]
        box = g.trimesh.primitives.Box(extents=start)
        assert g.np.allclose(box.primitive.extents,
                             start)
        assert g.np.allclose(box.extents,
                             start)
        if g.has_path:
            # check to see if outline function works
            assert g.np.allclose(box.as_outline().extents, start)

    def test_cyl_buffer(self):
        # test our inflation of cylinder primitives
        c = g.trimesh.primitives.Cylinder(
            radius=1.0,
            height=10.0,
            transform=g.trimesh.transformations.random_rotation_matrix())
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
            assert hasattr(primitive, 'transform')

            assert g.trimesh.util.is_shape(primitive.transform,
                                           (4, 4))

            if hasattr(primitive.primitive, 'center'):
                assert g.np.allclose(primitive.primitive.center,
                                     primitive.transform[:3, 3])


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
