try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import meshpy
    has_meshpy = True
except ImportError:
    g.log.warning('No meshpy! Not testing extrude primitives!')
    has_meshpy = False


class BooleanTest(g.unittest.TestCase):

    def setUp(self):
        self.primitives = []

        # do it with a flag in case there is more than one ImportError
        if has_meshpy:
            e = g.trimesh.primitives.Extrusion()
            e.primitive.polygon = g.trimesh.path.polygons.random_polygon()
            e.primitive.height = 1.0
            self.primitives.append(e)

            self.primitives.append(
                g.trimesh.primitives.Extrusion(
                    polygon=g.trimesh.path.polygons.random_polygon(),
                    height=293292.322))

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
            self.assertTrue(g.trimesh.util.is_shape(primitive.faces,
                                                    (-1, 3)))
            self.assertTrue(g.trimesh.util.is_shape(primitive.vertices,
                                                    (-1, 3)))

            self.assertTrue(primitive.volume > 0.0)
            self.assertTrue(primitive.area > 0.0)

            ratio = [primitive.volume / primitive.to_mesh().volume,
                     primitive.area / primitive.to_mesh().area]

            assert g.np.allclose(
                primitive.extents,
                primitive.to_mesh().extents)
            assert g.np.allclose(primitive.bounds, primitive.to_mesh().bounds)

            assert all(g.np.abs(i - 1) < 1e-2 for i in ratio)

            self.assertTrue(primitive.is_winding_consistent)
            self.assertTrue(primitive.is_watertight)

            # check that overload of dir worked
            self.assertTrue(
                len([i for i in dir(primitive.primitive) if not '_' in i]) > 0)

            if hasattr(primitive, 'direction'):
                self.assertTrue(primitive.direction.shape == (3,))

            centroid = primitive.centroid.copy()
            translation = [0, 0, 5]
            primitive.apply_translation(translation)

            # centroid should have translated correctly
            assert g.np.allclose(primitive.centroid - centroid, translation)

    def test_extrusion(self):
        if not has_meshpy:
            return

        polygon = g.Point([0, 0]).buffer(.5)
        e = g.trimesh.primitives.Extrusion(
            polygon=polygon,
            transform=g.trimesh.transformations.random_rotation_matrix())

        # will create an inflated version of the extrusion
        b = e.buffer(.1)

        assert b.volume > e.volume
        assert b.contains(e.vertices).all()

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


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
