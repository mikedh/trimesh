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

            self.primitives.append(g.trimesh.primitives.Extrusion(polygon=g.trimesh.path.polygons.random_polygon(),
                                                                  height=293292.322))

        self.primitives.append(g.trimesh.primitives.Sphere())
        self.primitives.append(g.trimesh.primitives.Sphere(center=[0, 0, 100],
                                                           radius=10.0,
                                                           subdivisions=5))
        self.primitives.append(g.trimesh.primitives.Box())
        self.primitives.append(g.trimesh.primitives.Box(center=[102.20, 0, 102.0],
                                                        extents=[29, 100, 1000]))

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

            self.assertTrue(primitive.is_winding_consistent)
            self.assertTrue(primitive.is_watertight)

            # check that overload of dir worked
            self.assertTrue(
                len([i for i in dir(primitive.primitive) if not '_' in i]) > 0)

            if hasattr(primitive, 'direction'):
                self.assertTrue(primitive.direction.shape == (3,))

    def test_extrusion(self):
        if not has_meshpy:
            return
        
        polygon = g.Point([0,0]).buffer(.5)
        e = g.trimesh.primitives.Extrusion(polygon=polygon,
            transform=g.trimesh.transformations.random_rotation_matrix())

        # will create an inflated version of the extrusion
        b = e.buffer(.1)

        self.assertTrue(b.volume > e.volume)
        self.assertTrue(b.contains(e.vertices).all())

                
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
