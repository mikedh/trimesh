try:
    from . import generic as g
except BaseException:
    import generic as g


def test_imc_sphere():
    for radius in [0.1, 1.0, 3.1459, 29.20]:
        m = g.trimesh.creation.icosphere(subdivisions=4, radius=radius)
        # the analytical value for a sphere
        reference = 4 * g.np.pi * radius
        assert g.np.isclose(m.integral_mean_curvature, reference, rtol=1e-3)


def test_imc_capsule():
    radius = 1.2
    for aspect_ratio in [0.0, 0.5, 1.0, 4.0, 100]:
        L = aspect_ratio * radius
        m = g.trimesh.creation.capsule(height=L, radius=radius, count=[64, 64])
        # the analytical value for a capsule
        reference = g.np.pi * (L + 4 * radius)
        assert g.np.isclose(m.integral_mean_curvature, reference, rtol=1e-3)


def test_imc_box(iterations: int = 4):
    extents = 1 - g.random((iterations, 3))
    for extent in extents:
        m = g.trimesh.creation.box(extents=extent)
        # only the right angle (=pi/2) edges contribute,
        # edges that lie on the faces of the box do not
        reference = 4 * extent.sum() * g.np.pi / 2 * 0.5
        assert g.np.isclose(m.integral_mean_curvature, reference, rtol=1e-3)


def test_imc_torus():
    r = 1.1
    for aspect_ratio in [1.1, 2.1, 4.0, 10.0]:
        R = r * aspect_ratio
        m = g.trimesh.creation.torus(
            major_radius=R, minor_radius=r, major_sections=100, minor_sections=100
        )
        reference = 2 * (g.np.pi**2) * R
        assert g.np.isclose(m.integral_mean_curvature, reference, rtol=1e-2)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()

    from pyinstrument import Profiler

    with Profiler() as P:
        test_imc_sphere()
        test_imc_box()
        test_imc_torus()
        test_imc_capsule()
    P.print()
