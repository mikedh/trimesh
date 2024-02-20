try:
    from . import generic as g
except BaseException:
    import generic as g


class ThicknessTest(g.unittest.TestCase):
    def test_sphere_thickness(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)

        # check that thickness measures are non-negative
        thickness = g.trimesh.proximity.thickness(
            mesh=m,
            points=samples,
            exterior=False,
            normals=m.face_normals[faces],
            method="max_sphere",
        )
        assert (thickness > -g.trimesh.tol.merge).all()

        # check thickness at a specific point
        point = g.np.array([[0.5, 0.0, 0.0]])
        point_thickness = g.trimesh.proximity.thickness(
            m, point, exterior=False, method="max_sphere"
        )

        # its a unit cube
        assert g.np.allclose(point_thickness, 1.0)

    def test_ray_thickness(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)

        # check that thickness and measures are non-negative
        thickness = g.trimesh.proximity.thickness(
            mesh=m,
            points=samples,
            exterior=False,
            normals=m.face_normals[faces],
            method="ray",
        )

        assert (thickness > -g.trimesh.tol.merge).all()

        # check thickness at a specific point
        point = g.np.array([[0.5, 0.0, 0.0]])
        point_thickness = g.trimesh.proximity.thickness(
            mesh=m, points=point, exterior=False, method="ray"
        )
        assert g.np.allclose(point_thickness, 1.0)

    def test_sphere_reach(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)

        # check that reach measures are infinite
        reach = g.trimesh.proximity.thickness(
            mesh=m,
            points=samples,
            exterior=True,
            normals=m.face_normals[faces],
            method="max_sphere",
        )
        assert g.np.isinf(reach).all()

    def test_ray_reach(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)

        # check that reach measures are infinite
        reach = g.trimesh.proximity.thickness(
            mesh=m,
            points=samples,
            exterior=True,
            normals=m.face_normals[faces],
            method="ray",
        )
        assert g.np.isinf(reach).all()

    def test_known(self):
        # an axis aligned plate part
        m = g.get_mesh("1002_tray_bottom.STL")

        # points on the surface
        samples = m.sample(1000)

        # compute thicknesses using sphere method
        rs = g.trimesh.proximity.thickness(mesh=m, points=samples, method="max_sphere")
        # compute thicknesses using pure ray tests
        ra = g.trimesh.proximity.thickness(mesh=m, points=samples, method="ray")
        # mesh is axis aligned plate so thickness is min extent
        truth = m.extents.min()

        assert g.np.isclose(g.np.median(rs), truth)
        assert g.np.isclose(g.np.median(ra), truth)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
