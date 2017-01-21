import generic as g


class SectionTest(g.unittest.TestCase):

    def setUp(self):
        self.mesh = g.get_mesh('featuretype.STL')

    def test_section(self):
        # this hits every case of section due to the geometry of this model
        step = .125
        z_levels = g.np.arange(start=self.mesh.bounds[0][2],
                               stop=self.mesh.bounds[1][2] + 2 * step,
                               step=step)

        for z in z_levels:
            plane_origin = [0, 0, z]
            plane_normal = [0, 0, 1]

            try:
                section = self.mesh.section(plane_origin=plane_origin,
                                            plane_normal=plane_normal)
            except ValueError:
                # section will raise a ValueError if the plane doesn't
                # intersect the mesh
                assert z > (self.mesh.bounds[1][
                            2] - g.trimesh.constants.tol.merge)

            planar, to_3D = section.to_planar()
            assert planar.is_closed
            assert (len(planar.polygons_full) > 0)


class PlaneLine(g.unittest.TestCase):

    def test_planes(self):
        count = 10
        z = g.np.linspace(-1, 1, count)

        plane_origins = g.np.column_stack((g.np.random.random((count, 2)), z))
        plane_normals = g.np.tile([0, 0, -1], (count, 1))

        line_origins = g.np.tile([0, 0, 0], (count, 1))
        line_directions = g.np.random.random((count, 3))

        i, valid = g.trimesh.intersections.planes_lines(plane_origins=plane_origins,
                                                        plane_normals=plane_normals,
                                                        line_origins=line_origins,
                                                        line_directions=line_directions)
        self.assertTrue(valid.all())
        self.assertTrue((g.np.abs(i[:, 2] - z) < g.tol.merge).all())

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
