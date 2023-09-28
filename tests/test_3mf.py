try:
    from . import generic as g
except BaseException:
    import generic as g


class MFTest(g.unittest.TestCase):
    def test_3MF(self):
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

    def test_units(self):
        # test our unit conversion function
        converter = g.trimesh.units.unit_conversion
        # these are the units listed in the 3MF spec as valid
        units = ["micron", "millimeter", "centimeter", "inch", "foot", "meter"]
        # check conversion factor for all valid 3MF units
        assert all(converter(u, "inches") > 1e-12 for u in units)

    def test_kwargs(self):
        # check if kwargs are properly passed to geometries
        s = g.get_mesh("P_XPM_0331_01.3mf")
        assert all(len(v.vertices) == 4 for v in s.geometry.values())

        s = g.get_mesh("P_XPM_0331_01.3mf", process=False)
        assert all(len(v.vertices) == 5 for v in s.geometry.values())

    def test_names(self):
        # check if two different objects with the same name are correctly
        # processed
        s = g.get_mesh("cube_and_sphere_same_name.3mf")
        assert len(s.geometry) == 2

    def test_roundtrip(self):
        if g.sys.version_info < (3, 6):
            g.log.warning("relies on > Python 3.5")
            return

        # test a scene round-tripped through the
        # 3MF exporter and importer
        s = g.get_mesh("cycloidal.3DXML")
        assert len(s.geometry) == 13

        # export and reload
        r = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(s.export(file_type="3mf")),
            file_type="3mf",
        )

        assert set(s.geometry.keys()) == set(r.geometry.keys())
        assert g.np.allclose(s.bounds, r.bounds)
        assert g.np.isclose(s.area, r.area, rtol=1e-3)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
