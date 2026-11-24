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
    # test a scene round-tripped through the
    # 3MF exporter and importer

    for name, count in [("cycloidal.3DXML", 13), ("pyramids.3mf", 6), ("nested.glb", 3)]:
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
        assert len(s.graph.nodes_geometry) == len(r.graph.nodes_geometry)

        if "3mf" in name.lower():
            # check partnumbers are preserved in the input 3mf files
            s_partnumbers = {e[1] for e in s.graph.to_edgelist()}
            r_partnumbers = {e[1] for e in r.graph.to_edgelist()}
            assert s_partnumbers == r_partnumbers, (s_partnumbers, r_partnumbers)


def test_metadata():
    import io
    import zipfile

    from lxml import etree

    # existing test models carry standard 3MF metadata
    feat = g.get_mesh("featuretype.3MF")
    assert feat.metadata["Title"] == "featuretype"
    assert feat.metadata["Designer"] == "mikedh"

    xpm = g.get_mesh("P_XPM_0331_01.3mf")
    assert xpm.metadata["Description"] == "3MF Test Case - Do not modify"
    assert "3MF Consortium" in xpm.metadata["Copyright"]

    # valid core 3MF keys round-trip; unknown keys and invalid types are skipped
    valid = {
        "Title": "Bracket & Mount <v2>",
        "Designer": "Jane Doe",
        "Description": "Test part",
        "LicenseTerms": "MIT",
        "CreationDate": "2026-10-22",
        "ModificationDate": "2026-11-24",
        "Application": "trimesh",
    }
    scene = g.trimesh.Scene(g.trimesh.creation.box())
    scene.metadata.update(valid)
    scene.metadata.update(
        {"CustomKey": "ignored", "Rating": None, "Copyright": 123, "file_name": "a.3mf"}
    )
    blob = scene.export(file_type="3mf")

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        root = etree.fromstring(z.read("3D/3dmodel.model"))
    tags = [etree.QName(c).localname for c in root]
    meta_els = [c for c in root if etree.QName(c).localname == "metadata"]
    assert {el.attrib["name"]: el.text for el in meta_els} == valid
    assert max(i for i, t in enumerate(tags) if t == "metadata") < tags.index("resources")

    reloaded = g.trimesh.load(
        file_obj=g.trimesh.util.wrap_as_stream(blob), file_type="3mf"
    )
    assert all(reloaded.metadata[k] == v for k, v in valid.items())
    assert all(
        k not in reloaded.metadata
        for k in ("CustomKey", "Rating", "Copyright", "file_name")
    )
    assert reloaded.metadata["units"] == "millimeter"

    # single Trimesh export preserves its metadata
    mesh = g.trimesh.creation.box()
    mesh.metadata.update({"Designer": "Jane Doe", "LicenseTerms": "CC-BY-4.0"})
    reloaded_mesh = g.trimesh.load(
        file_obj=g.trimesh.util.wrap_as_stream(mesh.export(file_type="3mf")),
        file_type="3mf",
    )
    assert reloaded_mesh.metadata["Designer"] == "Jane Doe"
    assert reloaded_mesh.metadata["LicenseTerms"] == "CC-BY-4.0"

    # object-level metadatagroup entries and non-standard names must not pollute metadata
    xml = b"""<?xml version="1.0" encoding="utf-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Title">outer</metadata>
  <metadata name="file_name">spoofed.3mf</metadata>
  <resources><object id="1" type="model">
    <metadatagroup><metadata name="Title">inner</metadata></metadatagroup>
    <mesh><vertices>
      <vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/>
    </vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh>
  </object></resources><build><item objectid="1"/></build>
</model>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("3D/3dmodel.model", xml)
    buf.seek(0)
    loaded = g.trimesh.load(file_obj=buf, file_type="3mf")
    assert loaded.metadata["Title"] == "outer"
    assert "file_name" not in loaded.metadata


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
