import generic as g


class UnitsTest(g.unittest.TestCase):

    def test_units(self):

        fake_units = 'blorbs'
        self.assertFalse(g.trimesh.units.validate(fake_units))

        m = g.get_mesh('featuretype.STL')
        self.assertTrue(m.units is None)

        m.units = 'in'
        self.assertTrue(m.units == 'in')

        extents_pre = m.extents
        m.convert_units('mm')
        scale = g.np.divide(m.extents, extents_pre)
        self.assertTrue(g.np.allclose(scale, 25.4))
        self.assertTrue(m.units == 'mm')


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
