try:
    from . import generic as g
except BaseException:
    import generic as g


def test_units():
    # make sure unit conversions fail for fake units
    fake_units = "blorbs"
    fake_units = "in"
    try:
        c = g.trimesh.units.unit_conversion("inches", fake_units)  # NOQA
        raise AssertionError()
    except BaseException:
        pass

    m = g.get_mesh("featuretype.STL")
    assert m.units is None

    m.units = "in"
    assert m.units == "in"

    extents_pre = m.extents
    m.convert_units("mm")
    scale = g.np.divide(m.extents, extents_pre)
    assert g.np.allclose(scale, 25.4)
    assert m.units == "mm"


def test_mm_hint():
    # get an STL that has "mm" in the name
    m = g.get_mesh("20mm-xyz-cube.stl")
    # convert it to meters
    m.convert_units("m")
    # should have picked the millimeters from the name
    assert g.np.allclose(m.extents, 0.02)


def test_conversion():
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


def test_path():
    p = g.get_mesh("2D/tray-easy1.dxf")
    # should be inches
    assert "in" in p.units
    extents_pre = p.extents
    p.convert_units("mm")
    # should have converted in -> mm 25.4
    # extents should scale exactly with unit conversion
    assert g.np.allclose(p.extents / extents_pre, 25.4, atol=0.01)


def test_keys():
    units = g.trimesh.units.keys()
    assert isinstance(units, set)
    assert "in" in units


def test_arbitrary():
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
    assert ac(0.01, g.trimesh.units.unit_conversion(current="0.01*m", desired="meters"))


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_mm_hint()
