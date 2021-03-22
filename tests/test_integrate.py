try:
    from . import generic as g
except BaseException:
    import generic as g


class IntegrateTest(g.unittest.TestCase):

    def test_integrate(self):
        try:
            from trimesh.integrate import symbolic_barycentric
            import sympy as sp
        except BaseException:
            g.log.warning('no sympy', exc_info=True)
            return

        m = g.get_mesh('featuretype.STL')

        integrator, expr = symbolic_barycentric('1')
        assert g.np.allclose(integrator(m).sum(), m.area)
        x, y, z = sp.symbols('x y z')
        functions = [x**2 + y**2, x + y + z]

        for f in functions:
            integrator, expr = symbolic_barycentric(f)
            integrator_p, expr_p = symbolic_barycentric(str(f))

            g.log.debug('expression %s was integrated to %s',
                        str(f),
                        str(expr))

            summed = integrator(m).sum()
            summed_p = integrator_p(m).sum()
            assert g.np.allclose(summed, summed_p)
            assert not g.np.allclose(summed, 0.0)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
