try:
    from . import generic as g
except BaseException:
    import generic as g


class UnitsTest(g.unittest.TestCase):
    def test_units(self):
        # make sure unit conversions fail for fake units
        fake_units = "blorbs"
        fake_units = "in"
        try:
            c = g.trimesh.units.unit_conversion("inches", fake_units)  # NOQA
            raise AssertionError()
        except BaseException:
            pass

        m = g.get_mesh("featuretype.STL")
        self.assertTrue(m.units is None)

        m.units = "in"
        self.assertTrue(m.units == "in")

        extents_pre = m.extents
        m.convert_units("mm")
        scale = g.np.divide(m.extents, extents_pre)
        self.assertTrue(g.np.allclose(scale, 25.4))
        self.assertTrue(m.units == "mm")

    def test_conversion(self):
        # test conversions on a multibody STL in a scene

        # a multibody STL with a unit hint in filename
        m = g.get_mesh("counter.unitsmm.STL")

        # nothing should be set
        assert m.units is None

        # split into watertight bodies
        s = g.trimesh.scene.split_scene(m)

        # save the extents
        extents_pre = s.extents

        # should extract units from file name without
        # raising a ValueError
        c = s.convert_units("in", guess=False)
        # should have converted mm -> in, 1/25.4
        # extents should scale exactly with unit conversion
        assert g.np.allclose(extents_pre / c.extents, 25.4, atol=0.01)

    def test_path(self):
        p = g.get_mesh("2D/tray-easy1.dxf")
        # should be inches
        assert "in" in p.units
        extents_pre = p.extents
        p.convert_units("mm")
        # should have converted in -> mm 25.4
        # extents should scale exactly with unit conversion
        assert g.np.allclose(p.extents / extents_pre, 25.4, atol=0.01)

    def test_keys(self):
        units = g.trimesh.units.keys()
        assert isinstance(units, set)
        assert "in" in units

    def test_arbitrary(self):
        ac = g.np.allclose
        to_inch = g.trimesh.units.to_inch

        # check whitespace
        assert ac(1.0, to_inch("in"))
        assert ac(1.0, to_inch("1.00000* in"))
        assert ac(1.0, to_inch("1.00    * in"))

        # check centimeter conversion
        assert ac(100, to_inch("m") / to_inch("0.01*m"))

        # if we are currently in centimeters and want to go to meters
        # it should be dividing it by 100
        assert ac(
            0.01, g.trimesh.units.unit_conversion(current="0.01*m", desired="meters")
        )


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
