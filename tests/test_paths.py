try:
    from . import generic as g
except BaseException:
    import generic as g


def test_discrete():
    for d in g.get_2D():
        # store hash before requesting passive functions
        hash_val = d.__hash__()

        g.check_path2D(d)

        # make sure various methods return
        # basically the same bounds
        atol = d.scale / 1000
        for dis, pa, pl in zip(d.discrete, d.paths, d.polygons_closed):
            # bounds of discrete version of path
            bd = g.np.array([g.np.min(dis, axis=0), g.np.max(dis, axis=0)])
            # bounds of polygon version of path
            bl = g.np.reshape(pl.bounds, (2, 2))
            # try bounds of included entities from path
            pad = g.np.vstack([d.entities[i].discrete(d.vertices) for i in pa])
            bp = g.np.array([g.np.min(pad, axis=0), g.np.max(pad, axis=0)])

            assert g.np.allclose(bd, bl, atol=atol)
            assert g.np.allclose(bl, bp, atol=atol)

        # run some checks
        g.check_path2D(d)

        # copying shouldn't touch original file
        copied = d.copy()

        # these operations shouldn't have mutated anything!
        assert d.__hash__() == hash_val

        # copy should have saved the metadata
        assert set(copied.metadata.keys()) == set(d.metadata.keys())

        # file_name should be populated, and if we have a DXF file
        # the layer field should be populated with layer names
        if d.source.file_name[-3:] == "dxf":
            assert len(d.layers) == len(d.entities)

        for path, verts in zip(d.paths, d.discrete):
            assert len(path) >= 1
            dists = g.np.sum((g.np.diff(verts, axis=0)) ** 2, axis=1) ** 0.5

            if not g.np.all(dists > g.tol_path.zero):
                raise ValueError("{} had zero distance in discrete!", d.source.file_name)

            circuit_dist = g.np.linalg.norm(verts[0] - verts[-1])
            circuit_test = circuit_dist < g.tol_path.merge
            if not circuit_test:
                g.log.error(
                    "On file %s First and last vertex distance %f",
                    d.source.file_name,
                    circuit_dist,
                )
            assert circuit_test

            is_ccw = g.trimesh.path.util.is_ccw(verts)
            if not is_ccw:
                g.log.error("discrete %s not ccw!", d.source.file_name)

        for i in range(len(d.paths)):
            assert d.polygons_closed[i].is_valid
            assert d.polygons_closed[i].area > g.tol_path.zero
        export_dict = d.export(file_type="dict")
        to_dict = d.to_dict()
        assert isinstance(to_dict, dict)
        assert isinstance(export_dict, dict)
        assert len(to_dict) == len(export_dict)

        export_svg = d.export(file_type="svg")  # NOQA
        simple = d.simplify()  # NOQA
        split = d.split()
        g.log.info(
            "Split %s into %d bodies, checking identifiers",
            d.source.file_name,
            len(split),
        )
        for body in split:
            _ = body.identifier

        if len(d.root) == 1:
            d.apply_obb()

        # store the X values of bounds
        ori = d.bounds.copy()
        # apply a translation
        d.apply_translation([10, 0])
        # X should have translated by 10.0
        assert g.np.allclose(d.bounds[:, 0] - 10, ori[:, 0])
        # Y should not have moved
        assert g.np.allclose(d.bounds[:, 1], ori[:, 1])

        if len(d.polygons_full) > 0 and len(d.vertices) < 150:
            g.log.info("Checking medial axis on %s", d.source.file_name)
            m = d.medial_axis()
            assert len(m.entities) > 0

        # shouldn't crash
        d.fill_gaps()

        # transform to first quadrant
        d.rezero()
        # run process manually
        d.process()


def test_path():
    path = g.trimesh.path.Path3D(
        entities=[g.trimesh.path.entities.Line(points=[0, 1, 2])],
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0]],
        vertex_attributes={"_test": g.np.array([1, 2, 3], dtype=g.np.float32)},
    )

    glb_data = g.to_glb_bytes(path)
    loaded_scene = g.from_glb_bytes(glb_data)

    loaded_path = loaded_scene.geometry["geometry_0"]
    loaded_path.process()

    assert isinstance(loaded_path, g.trimesh.path.Path3D)

    assert loaded_path.vertices.shape == (3, 3)
    assert len(loaded_path.entities) == 1
    assert len(loaded_path.vertex_attributes["_test"]) == 3


def test_path_to_gltf_with_line():
    path = g.trimesh.path.Path3D(
        entities=[
            g.trimesh.path.entities.Line(points=[0, 1, 2]),
            g.trimesh.path.entities.Line(points=[3, 4, 5]),
        ],
        vertices=g.np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 2],
            ]
        ),
    )
    path.vertex_attributes = {"_test": g.np.array([0, 0, 0, 1, 1, 1], dtype=g.np.float32)}

    tree, _ = g.trimesh.exchange.gltf._create_gltf_structure(g.trimesh.Scene([path]))
    # should have included the attribute by name
    assert set(tree["meshes"][0]["primitives"][0]["attributes"].keys()) == {
        "POSITION",
        "_test",
    }

    assert len(tree["accessors"]) == 2
    acc = tree["accessors"]
    assert acc[0]["count"] == acc[1]["count"]


def test_path_to_gltf_with_arc():
    path = g.trimesh.path.Path3D(
        entities=[
            g.trimesh.path.entities.Line(points=[0, 1, 2]),
            g.trimesh.path.entities.Arc(points=[3, 4, 5]),
        ],
        vertices=g.np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 2],
            ]
        ),
    )
    path.vertex_attributes = {"_test": g.np.array([0, 0, 0, 1, 1, 1], dtype=g.np.float32)}

    tree, _ = g.trimesh.exchange.gltf._create_gltf_structure(g.trimesh.Scene([path]))

    # should have skipped the attribute rather than exporting a broken one
    assert set(tree["meshes"][0]["primitives"][0]["attributes"].keys()) == {"POSITION"}


def test_poly():
    p = g.get_mesh("2D/LM2.dxf")
    assert p.is_closed

    # one of the lines should be a polyline
    assert any(
        len(e.points) > 2
        for e in p.entities
        if isinstance(e, g.trimesh.path.entities.Line)
    )

    # layers should match entity count
    assert len(p.layers) == len(p.entities)
    assert len(set(p.layers)) > 1

    # check copy with layer specified
    unique = set(p.layers)
    assert len(unique) > 1
    for u in unique:
        a = p.copy(layers={u})
        assert len(a.entities) > 0
        assert p.layers.count(u) == len(a.entities)
        assert len(a.entities) < len(p.layers)
        assert all(e.layer == u for e in a.entities)

    # try exploding
    count = len(p.entities)

    p.explode()
    # explode should have created new entities
    assert len(p.entities) > count
    # explode should have added some new layers
    assert len(p.entities) == len(p.layers)
    # all line segments should have two points now
    assert all(
        len(i.points) == 2
        for i in p.entities
        if isinstance(i, g.trimesh.path.entities.Line)
    )
    # should still be closed
    assert p.is_closed
    # chop off the last entity
    p.entities = p.entities[:-1]
    # should no longer be closed
    assert not p.is_closed

    # fill gaps of any distance
    p.fill_gaps(g.np.inf)
    # should have fixed this puppy
    assert p.is_closed

    # remove 2 short edges using remove_entity()
    count = len(p.entities)
    p.remove_entities([count - 1, count - 6])
    assert not p.is_closed
    p.fill_gaps(2)
    assert p.is_closed


def test_text():
    """
    Do some checks on Text entities
    """
    p = g.get_mesh("2D/LM2.dxf")
    p.explode()
    # get some text entities
    text = [e for e in p.entities if isinstance(e, g.trimesh.path.entities.Text)]
    assert len(text) > 1

    # loop through each of them
    for t in text:
        # a spurious error we were seeing in CI
        if g.trimesh.util.is_instance_named(t, "Line"):
            raise ValueError(
                "type bases:", [i.__name__ for i in g.trimesh.util.type_bases(t)]
            )
    # make sure this doesn't crash with text entities
    g.trimesh.rendering.convert_to_vertexlist(p)


def test_empty():
    # make sure empty paths perform as expected

    p = g.trimesh.path.Path2D()
    assert p.is_empty
    p = g.trimesh.path.Path3D()
    assert p.is_empty

    p = g.trimesh.load_path([[[0, 0, 1], [1, 0, 1]]])
    assert len(p.entities) == 1
    assert not p.is_empty

    b = p.to_planar()[0]
    assert len(b.entities) == 1
    assert not b.is_empty


def test_color():
    p = g.get_mesh("2D/wrench.dxf")
    # make sure we have entities
    assert len(p.entities) > 0
    # make sure shape of colors is correct
    assert all(e.color is None for e in p.entities)
    assert p.colors is None

    color = [255, 0, 0, 255]
    # assign a color to the entity
    p.entities[0].color = color
    # make sure this is reflected in the path color
    assert g.np.allclose(p.colors[0], color)
    assert p.colors.shape == (len(p.entities), 4)

    p.colors = g.np.array([100, 100, 100] * len(p.entities), dtype=g.np.uint8).reshape(
        (-1, 3)
    )
    assert g.np.allclose(p.colors[0], [100, 100, 100, 255])


def test_dangling():
    p = g.get_mesh("2D/wrench.dxf")
    assert len(p.dangling) == 0

    b = g.get_mesh("2D/loose.dxf")
    assert len(b.dangling) == 1
    assert len(b.polygons_full) == 1


def test_plot():
    try:
        # only run these if matplotlib is installed
        import matplotlib.pyplot  # NOQA
    except BaseException:
        g.log.debug("skipping `matplotlib.pyplot` tests")
        return

    p = g.get_mesh("2D/wrench.dxf")

    # see if the logic crashes
    p.plot_entities(show=False)
    p.plot_discrete(show=False)


def test_split():
    for fn in [
        "2D/ChuteHolderPrint.DXF",
        "2D/tray-easy1.dxf",
        "2D/sliding-base.dxf",
        "2D/wrench.dxf",
        "2D/spline_1.dxf",
    ]:
        p = g.get_mesh(fn)

        # make sure something was loaded
        assert len(p.root) > 0

        # split by connected
        split = p.split()

        # make sure split parts have same area as source
        assert g.np.isclose(p.area, sum(i.area for i in split))
        # make sure concatenation doesn't break that
        assert g.np.isclose(p.area, g.np.sum(split).area)

        # check that cache didn't screw things up
        for s in split:
            assert len(s.root) == 1
            assert len(s.path_valid) == len(s.paths)
            assert len(s.paths) == len(s.discrete)
            assert s.path_valid.sum() == len(s.polygons_closed)
            g.check_path2D(s)


def test_section():
    mesh = g.get_mesh("tube.obj")

    # check the CCW correctness with a normal in both directions
    for sign in [1.0, -1.0]:
        # get a cross section of the tube
        section = mesh.section(
            plane_origin=mesh.center_mass, plane_normal=[0.0, sign, 0.0]
        )

        # Path3D -> Path2D
        planar, _T = section.to_planar()

        # tube should have one closed polygon
        assert len(planar.polygons_full) == 1
        polygon = planar.polygons_full[0]
        # closed polygon should have one interior
        assert len(polygon.interiors) == 1

        # the exterior SHOULD be counterclockwise
        assert g.trimesh.path.util.is_ccw(polygon.exterior.coords)
        # the interior should NOT be counterclockwise
        assert not g.trimesh.path.util.is_ccw(polygon.interiors[0].coords)

        # should be a valid Path2D
        g.check_path2D(planar)


def test_multiplane():
    # check to make sure we're applying `tol_path.merge_digits` for line segments
    vertices = g.np.array(
        [
            [19.402250931139097, -14.88787016674277, 64.0],
            [20.03318396099334, -14.02738654377374, 64.0],
            [19.402250931139097, -14.88787016674277, 0.0],
            [20.03318396099334, -14.02738654377374, 0.0],
            [21, -16.5, 32],
        ]
    )
    faces = g.np.array([[1, 3, 0], [0, 3, 2], [1, 0, 4], [0, 2, 4], [3, 1, 4], [2, 3, 4]])
    z_layer_centers = g.np.array([3, 3.0283334255218506])
    m = g.trimesh.Trimesh(vertices, faces)
    r = m.section_multiplane([0, 0, 0], [0, 0, 1], z_layer_centers)
    assert len(r) == 2
    assert all(i.is_closed for i in r)


def test_svg_arc_snap():
    s = """<svg xmlns="http://www.w3.org/2000/svg" width="97.5mm" height="35mm" viewBox="-17.5 0 97.5 35">

<path d="M 0 0 h 80 v 35 h -80 a 17.5 17.5 0 0 1 0 -35 z" fill="none" stroke="black" stroke-width="0.1" />
</svg>"""

    p = g.trimesh.load_path(g.trimesh.util.wrap_as_stream(s), file_type="svg")

    # should be one closed path
    assert len(p.root) == 1
    # should be exactly closed with a snap
    assert g.np.equal(*p.discrete[0][[0, -1]]).all()
    # the extrusion should be a volume
    assert p.extrude(10.0).is_volume


def test_path_dict_export():
    x, y = 2, 3

    # define incorrectly
    as_dict = {
        "entities": [
            {"type": "Line", "points": [0, 1, 2, 3, 0], "closed": False},
        ],
        "vertices": [
            [-x / 2, y / 2],
            [x / 2, y / 2],
            [x / 2, -y / 2],
            [-x / 2, -y / 2],
        ],
    }

    # try loading with the incorrect `closed` value for the line
    path = g.trimesh.path.exchange.load.load_path(
        g.trimesh.path.exchange.misc.dict_to_path(as_dict)
    )

    # export to a dict again
    exp = path.export(file_type="dict")
    assert len(exp["entities"]) == 1
    assert "closed" not in exp["entities"][0]

    # do a roundtrip
    rt = g.trimesh.path.exchange.load.load_path(
        g.trimesh.path.exchange.misc.dict_to_path(exp)
    )

    # should have the same area
    assert g.np.isclose(rt.area, path.area)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    # g.unittest.main()

    # test_path_to_gltf_with_arc()
    # test_path_dict_export()
    test_poly()
