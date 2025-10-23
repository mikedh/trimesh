try:
    from . import generic as g
except BaseException:
    import generic as g


def test_3MF():
    # an assembly with instancing
    s = g.get_mesh("counterXP.3MF")

    # should be 2 unique meshes
    assert len(s.geometry) == 2
    # should be 6 instances around the scene
    assert len(s.graph.nodes_geometry) == 6
    assert all(m.is_volume for m in s.geometry.values())

    # a single body 3MF assembly
    s = g.get_mesh("featuretype.3MF")
    # should be 2 unique meshes
    assert len(s.geometry) == 1
    # should be 6 instances around the scene
    assert len(s.graph.nodes_geometry) == 1


def test_units():
    # test our unit conversion function
    converter = g.trimesh.units.unit_conversion
    # these are the units listed in the 3MF spec as valid
    units = ["micron", "millimeter", "centimeter", "inch", "foot", "meter"]
    # check conversion factor for all valid 3MF units
    assert all(converter(u, "inches") > 1e-12 for u in units)


def test_more_units():
    # make sure that units survive down to the mesh
    s = g.trimesh.load_scene(g.os.path.join(g.dir_models, "featuretype.3MF"))
    m = next(iter(s.geometry.values()))
    assert m.units is not None, m.metadata


def test_kwargs():
    # check if kwargs are properly passed to geometries
    s = g.get_mesh("P_XPM_0331_01.3mf")
    assert all(len(v.vertices) == 4 for v in s.geometry.values())

    s = g.get_mesh("P_XPM_0331_01.3mf", process=False)
    assert all(len(v.vertices) == 5 for v in s.geometry.values())


def test_names():
    # check if two different objects with the same name are correctly
    # processed
    s = g.get_mesh("cube_and_sphere_same_name.3mf")
    assert len(s.geometry) == 2


def test_roundtrip():
    if g.sys.version_info < (3, 6):
        g.log.warning("relies on > Python 3.5")
        return

    # test a scene round-tripped through the
    # 3MF exporter and importer

    for name, count in [("cycloidal.3DXML", 13), ("pyramids.3mf", 6)]:
        s = g.get_mesh(name)
        assert len(s.geometry) == count

        # export and reload
        r = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(s.export(file_type="3mf")),
            file_type="3mf",
        )

        assert set(s.geometry.keys()) == set(r.geometry.keys()), (
            s.geometry.keys(),
            r.geometry.keys(),
        )
        assert g.np.allclose(s.bounds, r.bounds)
        assert g.np.isclose(s.area, r.area, rtol=1e-3)

        if "3mf" in name.lower():
            # check partnumbers are preserved in the input 3mf files
            s_partnumbers = {e[1] for e in s.graph.to_edgelist()}
            r_partnumbers = {e[1] for e in r.graph.to_edgelist()}
            assert s_partnumbers == r_partnumbers, (s_partnumbers, r_partnumbers)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
