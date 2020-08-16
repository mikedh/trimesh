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
                assert len(section.centroid) == 3

                planar, to_3D = section.to_planar()
                assert planar.is_closed
                assert (len(planar.polygons_full) > 0)
                assert len(planar.centroid) == 2

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

    def test_multi_index(self):
        # make sure returned face indexes on a section are correct
        mesh = g.trimesh.creation.box()
        normal = g.np.array([0, 0, 1])
        # point for plane origin
        origin = mesh.center_mass
        # pick some offsets from the plane
        heights = g.np.linspace(0, 0.1, 2)
        # compute the multiple crosss sections with `section_multiplane`
        multi = mesh.section_multiplane(origin, normal, heights)
        # get the face indexes this should have hit by checking the normal
        idx = set(g.np.nonzero(g.np.isclose(
            g.np.dot(mesh.face_normals, normal), 0, atol=1e-4))[0])
        # make sure all indexes are set correctly
        assert all(set(m.metadata['face_index'] == idx)
                   for m in multi)


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


class SliceTest(g.unittest.TestCase):

    def test_slice(self):
        mesh = g.trimesh.creation.box()

        # Cut corner off of box and make sure the bounds and number of faces is correct
        # Tests new triangles, but not new quads or triangles contained entirely
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = mesh.bounds[1]

        sliced = mesh.slice_plane(plane_origin=plane_origin,
                                  plane_normal=plane_normal)

        assert g.np.isclose(sliced.bounds[0], mesh.bounds[1] - 0.15).all()
        assert g.np.isclose(sliced.bounds[1], mesh.bounds[1]).all()
        assert len(sliced.faces) == 5

        # Cut top off of box and make sure bounds and number of faces is correct
        # Tests new quads and entirely contained triangles
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = g.np.array([0, 0, 1])

        sliced = mesh.slice_plane(plane_origin=plane_origin,
                                  plane_normal=plane_normal)

        assert g.np.isclose(
            sliced.bounds[0], mesh.bounds[0] + g.np.array([0, 0, 0.95])).all()
        assert g.np.isclose(sliced.bounds[1], mesh.bounds[1]).all()
        assert len(sliced.faces) == 14

        # non- watertight more complex mesh
        bunny = g.get_mesh('bunny.ply')

        origin = bunny.bounds.mean(axis=0)
        normal = g.trimesh.unitize([1, 1, 2])

        sliced = bunny.slice_plane(plane_origin=origin,
                                   plane_normal=normal)
        assert len(sliced.faces) > 0

        # check the projections manually
        dot = g.np.dot(normal, (sliced.vertices - origin).T)
        # should be lots of stuff at the plane and nothing behind
        assert g.np.isclose(dot.min(), 0.0)

        # Cut part of top off of box with multiple planes at once
        # and make sure bounds and number of faces is correct
        plane_origins = [mesh.bounds[1] - 0.05, mesh.bounds[1] - 0.05]
        plane_normals = [g.np.array([0, 0, 1]), g.np.array([0, 1, 0])]

        sliced = mesh.slice_plane(plane_origin=plane_origins,
                                  plane_normal=plane_normals)

        assert g.np.isclose(
            sliced.bounds[0], mesh.bounds[0] + g.np.array([0, 0.95, 0.95])).all()
        assert g.np.isclose(sliced.bounds[1], mesh.bounds[1]).all()
        assert len(sliced.faces) == 11

        # Try with more complicated mesh and make sure we get correct projections
        # and some faces
        origins = [bunny.bounds.mean(axis=0), bunny.bounds.mean(
            axis=0) + 0.01 * g.trimesh.unitize([1, 1, 2])]
        normals = [g.trimesh.unitize([1, 1, 2]), -g.trimesh.unitize([1, 1, 2])]

        sliced = bunny.slice_plane(plane_origin=origins,
                                   plane_normal=normals)
        assert len(sliced.faces) > 0

        for o, n in zip(origins, normals):
            # check the projections manually
            dot = g.np.dot(n, (sliced.vertices - o).T)
            # should be lots of stuff at the plane and nothing behind
            assert g.np.isclose(dot.min(), 0.0)

        # Test cap on more complicated watertight mesh to make sure the
        # resulting mesh is still watertight and slice is correct
        featuretype = g.get_mesh('featuretype.STL')

        origins = [featuretype.center_mass, featuretype.center_mass
                   + 0.01 * g.trimesh.unitize([1, 0, 2])]
        normals = [g.trimesh.unitize([1, 1, 1]), g.trimesh.unitize([1, 2, 3])]

    def test_cap(self):

        try:
            from triangle import triangulate  # NOQA
        except BaseException as E:
            if g.all_dep:
                raise E
            else:
                return

        mesh = g.trimesh.creation.box()

        # Cut corner off of box and make sure the bounds and number of faces is correct
        # Tests new triangles, but not new quads or triangles contained entirely
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = mesh.bounds[1]

        # Same test with capping (should only add three more triangles)
        sliced_capped = mesh.slice_plane(plane_origin=plane_origin,
                                         plane_normal=plane_normal,
                                         cap=True)

        assert len(sliced_capped.faces) == 8
        assert g.np.isclose(sliced_capped.bounds[0], mesh.bounds[1] - 0.15).all()
        assert g.np.isclose(sliced_capped.bounds[1], mesh.bounds[1]).all()
        assert sliced_capped.is_watertight

        # Cut top off of box and make sure bounds and number of faces is correct
        # Tests new quads and entirely contained triangles
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = g.np.array([0, 0, 1])

        # Same test with capping (should only add six triangles)
        sliced_capped = mesh.slice_plane(plane_origin=plane_origin,
                                         plane_normal=plane_normal,
                                         cap=True)

        assert len(sliced_capped.faces) == 20
        assert g.np.isclose(
            sliced_capped.bounds[0], mesh.bounds[0] + g.np.array([0, 0, 0.95])).all()
        assert g.np.isclose(sliced_capped.bounds[1], mesh.bounds[1]).all()
        assert sliced_capped.is_watertight

        # Cut part of top off of box with multiple planes at once
        # and make sure bounds and number of faces is correct
        plane_origins = [mesh.bounds[1] - 0.05, mesh.bounds[1] - 0.05]
        plane_normals = [g.np.array([0, 0, 1]), g.np.array([0, 1, 0])]

        # Test cap for multiple slices to check watertightness
        # (should add nine triangles)
        sliced_capped = mesh.slice_plane(plane_origin=plane_origins,
                                         plane_normal=plane_normals,
                                         cap=True)
        assert g.np.isclose(
            sliced_capped.bounds[0], mesh.bounds[0] + g.np.array([0, 0.95, 0.95])).all()
        assert g.np.isclose(sliced_capped.bounds[1], mesh.bounds[1]).all()
        assert sliced_capped.is_watertight

        # Test cap on more complicated watertight mesh to make sure the
        # resulting mesh is still watertight and slice is correct
        featuretype = g.get_mesh('featuretype.STL')

        origins = [featuretype.center_mass, featuretype.center_mass
                   + 0.01 * g.trimesh.unitize([1, 0, 2])]
        normals = [g.trimesh.unitize([1, 0, 1]), g.trimesh.unitize([1, 0, 0])]

        sliced_capped = featuretype.slice_plane(
            plane_origin=origins,
            plane_normal=normals,
            cap=True)
        assert len(sliced_capped.faces) > 0

        for o, n in zip(origins, normals):
            # check the projections manually
            dot = g.np.dot(n, (sliced_capped.vertices - o).T)
            # should be lots of stuff at the plane and nothing behind
            assert g.np.isclose(dot.min(), 0.0)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
