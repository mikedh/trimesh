import generic as g

class SectionTest(g.unittest.TestCase):
    def setUp(self):
        self.mesh = g.get_mesh('featuretype.STL')

    def test_section(self):
        # this hits every case of section due to the geometry of this model
        step = .125
        z_levels = g.np.arange(start = self.mesh.bounds[0][2],
                               stop  = self.mesh.bounds[1][2] + 2*step,
                               step  = step)

        for z in z_levels:
            plane_origin = [0,0,z]
            plane_normal = [0,0,1]

            try:
                section = self.mesh.section(plane_origin = plane_origin,
                                            plane_normal = plane_normal)
            except ValueError:
                # section will raise a ValueError if the plane doesn't intersect the mesh
                assert z > (self.mesh.bounds[1][2] - g.trimesh.constants.tol.merge)

            planar, to_3D = section.to_planar()
            assert planar.is_closed
            assert (len(planar.polygons_full) > 0)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()

