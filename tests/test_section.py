try:
    from . import generic as g
except BaseException:
    import generic as g


class SectionTest(g.unittest.TestCase):

    def test_section(self):
        mesh = g.get_mesh('featuretype.STL')
        # this hits many edge cases
        step = .125
        # step through mesh
        z_levels = g.np.arange(start=mesh.bounds[0][2],
                               stop=mesh.bounds[1][2] + 2 * step,
                               step=step)
        # randomly order Z so first level is probably not zero
        z_levels = g.np.random.permutation(z_levels)

        # rotate around so we're not just testing XY parallel planes
        for angle in [0.0, g.np.radians(1.0), g.np.radians(11.11232)]:
            # arbitrary test rotation axis
            axis = g.trimesh.unitize([1.0, 2.0, .32343])
            # rotate plane around axis to test along non- parallel planes
            base = g.trimesh.transformations.rotation_matrix(angle=angle,
                                                             direction=axis)
            # to return to parallel to XY plane
            base_inv = g.np.linalg.inv(base)

            # transform normal to be slightly rotated
            plane_normal = g.trimesh.transform_points([[0, 0, 1.0]],
                                                      base,
                                                      translate=False)[0]

            # store Path2D and Path3D results
            sections = [None] * len(z_levels)
            sections_3D = [None] * len(z_levels)

            for index, z in enumerate(z_levels):
                # move origin into rotated frame
                plane_origin = [0, 0, z]
                plane_origin = g.trimesh.transform_points(
                    [plane_origin], base)[0]

                section = mesh.section(plane_origin=plane_origin,
                                       plane_normal=plane_normal)

                if section is None:
                    # section will return None if the plane doesn't
                    # intersect the mesh
                    # assert z > (mesh.bounds[1][2] -
                    #            g.trimesh.constants.tol.merge)
                    continue

                # move back
                section.apply_transform(base_inv)

                # Z should be in plane frame
                assert g.np.allclose(section.vertices[:, 2], z)

                planar, to_3D = section.to_planar()
                assert planar.is_closed
                assert (len(planar.polygons_full) > 0)
                sections[index] = planar
                sections_3D[index] = section

            # try getting the sections as Path2D through
            # the multiplane method
            paths = mesh.section_multiplane(plane_origin=[0, 0, 0],
                                            plane_normal=plane_normal,
                                            heights=z_levels)

            # call the multiplane method directly
            lines, faces, T = g.trimesh.intersections.mesh_multiplane(
                mesh=mesh,
                plane_origin=[0, 0, 0],
                plane_normal=plane_normal,
                heights=z_levels)

            # make sure various methods return the same results
            for index in range(len(z_levels)):
                if sections[index] is None:
                    assert len(lines[index]) == 0
                    # make sure mesh.multipath_section is the same
                    assert paths[index] is None
                    continue
                # reconstruct path from line segments
                rc = g.trimesh.load_path(lines[index])

                assert g.np.isclose(rc.area, sections[index].area)
                assert g.np.isclose(rc.area, paths[index].area)

                # send Path2D back to 3D using the transform returned by
                # section
                back_3D = paths[index].to_3D(paths[index].metadata['to_3D'])
                # move to parallel test plane
                back_3D.apply_transform(base_inv)

                # make sure all vertices have constant Z
                assert back_3D.vertices[:, 2].ptp() < 1e-8
                assert sections_3D[index].vertices[:, 2].ptp() < 1e-8

                # make sure reconstructed 3D section is at right height
                assert g.np.isclose(back_3D.vertices[:, 2].mean(),
                                    sections_3D[index].vertices[:, 2].mean())

                # make sure reconstruction is at z of frame
                assert g.np.isclose(back_3D.vertices[:, 2].mean(),
                                    z_levels[index])


class PlaneLine(g.unittest.TestCase):

    def test_planes(self):
        count = 10
        z = g.np.linspace(-1, 1, count)

        plane_origins = g.np.column_stack((
            g.np.random.random((count, 2)), z))
        plane_normals = g.np.tile([0, 0, -1], (count, 1))

        line_origins = g.np.tile([0, 0, 0], (count, 1))
        line_directions = g.np.random.random((count, 3))

        i, valid = g.trimesh.intersections.planes_lines(
            plane_origins=plane_origins,
            plane_normals=plane_normals,
            line_origins=line_origins,
            line_directions=line_directions)
        assert valid.all()
        assert (g.np.abs(i[:, 2] - z) < g.tol.merge).all()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
