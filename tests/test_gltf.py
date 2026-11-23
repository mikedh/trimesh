try:
    from . import generic as g
except BaseException:
    import generic as g

validate_glb = g.validate_glb
load_kwargs = g.trimesh.exchange.load._load_kwargs


def world_root_tree():
    # minimal single-file glTF whose ROOT node is named "world"
    # and carries a transform — the shape external tools produce
    indices = g.np.array([0, 1, 2], dtype=g.np.uint32).tobytes()
    uri = "data:application/octet-stream;base64," + g.base64.b64encode(indices).decode(
        "utf-8"
    )
    tree = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "world", "translation": [10.0, 0.0, 0.0], "children": [1]},
            {"name": "geom", "mesh": 0},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "accessors": [
            {"componentType": 5126, "count": 3, "type": "VEC3"},
            {"componentType": 5125, "count": 3, "type": "SCALAR", "bufferView": 0},
        ],
        "bufferViews": [{"buffer": 0, "byteLength": 12, "byteOffset": 0}],
        "buffers": [{"byteLength": 12, "uri": uri}],
    }
    return g.trimesh.util.wrap_as_stream(g.json.dumps(tree).encode("utf-8"))


class GLTFTest(g.unittest.TestCase):
    def test_duck(self):
        scene = g.get_mesh("Duck.glb", process=False)

        # should have one mesh
        assert len(scene.geometry) == 1

        # get the mesh
        geom = next(iter(scene.geometry.values()))

        # vertex normals should have been loaded
        assert "vertex_normals" in geom._cache.cache

        # should not be watertight
        assert not geom.is_volume
        # make sure export doesn't crash
        export = scene.export(file_type="glb")
        validate_glb(export, "Duck.glb")

        # check a roundtrip
        reloaded = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")
        # make basic assertions
        g.scene_equal(scene, reloaded)

        # if we merge ugly it should now be watertight
        geom.merge_vertices(merge_tex=True, merge_norm=True)
        assert geom.is_volume

    def test_strips(self):
        a = g.get_mesh("mode5.gltf")
        assert len(a.geometry) > 0

        b = g.get_mesh("mode5.gltf", merge_primitives=True)
        assert len(b.geometry) > 0

    def test_buffer_dedupe(self):
        scene = g.trimesh.Scene()
        box_1 = g.trimesh.creation.box()
        box_2 = g.trimesh.creation.box()
        box_3 = g.trimesh.creation.box()
        box_3.visual.face_colors = [0, 255, 0, 255]

        tm = g.trimesh.transformations.translation_matrix
        scene.add_geometry(box_1, "box_1", transform=tm((1, 1, 1)))
        scene.add_geometry(box_2, "box_2", transform=tm((-1, -1, -1)))
        scene.add_geometry(box_3, "box_3", transform=tm((-1, 20, -1)))
        a = g.json.loads(scene.export(file_type="gltf")["model.gltf"].decode("utf-8"))
        assert len(a["buffers"]) <= 3

    def test_unknown_chunk(self):
        # the GLTF spec requires loaders to ignore chunks with unknown
        # types following the JSON and BIN chunks
        m = g.get_mesh("fuze.obj")
        export = m.export(file_type="glb")
        # append a chunk with an unknown type and update the file length
        chunk = g.np.array([8, 0x41503243], dtype="<u4").tobytes() + b"\x00" * 8
        data = bytearray(export + chunk)
        data[8:12] = g.np.array([len(data)], dtype="<u4").tobytes()

        r = g.trimesh.load(g.trimesh.util.wrap_as_stream(bytes(data)), file_type="glb")
        assert g.np.isclose(r.area, m.area)

    def test_skip_materials(self):
        # load textured PLY
        mesh = g.get_mesh("fuze.ply")
        g.check_fuze(mesh)

        # load as GLB
        export = mesh.export(file_type="glb", unitize_normals=True)
        validate_glb(export)
        mesh_glb = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(export),
            file_type="glb",
            force="mesh",
            skip_materials=True,
        )

        # visuals should not be present
        assert not mesh_glb.visual.defined

    def test_tex_export(self):
        # load textured PLY
        mesh = g.get_mesh("fuze.ply")
        assert hasattr(mesh.visual, "uv")

        # make sure export as GLB doesn't crash on scenes
        export = mesh.scene().export(file_type="glb", unitize_normals=True)
        validate_glb(export, "fuze.ply")
        # make sure it works on meshes
        export = mesh.export(file_type="glb", unitize_normals=True)
        validate_glb(export, "fuze.ply")

    def test_cesium(self):
        # A GLTF with a multi- primitive mesh

        s = g.get_mesh("CesiumMilkTruck.glb")
        # should be one Trimesh object per GLTF "primitive"
        assert len(s.geometry) == 4
        # every geometry displayed once, except wheels twice
        assert len(s.graph.nodes_geometry) == 5

        # make sure export doesn't crash
        export = s.export(file_type="glb")
        validate_glb(export)

        reloaded = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")
        # make basic assertions
        g.scene_equal(s, reloaded)

    def test_alphamode(self):
        # A GLTF with combinations of AlphaMode and AlphaCutoff
        s = g.get_mesh("AlphaBlendModeTest.glb")
        # should be 5 test geometries
        assert len([geom for geom in s.geometry if geom.startswith("Test")]) == 5
        assert s.geometry["TestCutoffDefaultMesh"].visual.material.alphaMode == "MASK"
        assert s.geometry["TestCutoff25Mesh"].visual.material.alphaMode == "MASK"
        assert s.geometry["TestCutoff25Mesh"].visual.material.alphaCutoff == 0.25
        assert s.geometry["TestCutoff75Mesh"].visual.material.alphaMode == "MASK"
        assert s.geometry["TestCutoff75Mesh"].visual.material.alphaCutoff == 0.75
        assert s.geometry["TestBlendMesh"].visual.material.alphaMode == "BLEND"
        # defaults OPAQUE
        assert s.geometry["TestOpaqueMesh"].visual.material.alphaMode is None

        export = s.export(file_type="glb")
        validate_glb(export)

        # roundtrip it
        rs = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")

        # make basic assertions
        g.scene_equal(s, rs)

        # make sure export keeps alpha modes
        # should be the same
        assert len([geom for geom in rs.geometry if geom.startswith("Test")]) == 5
        assert rs.geometry["TestCutoffDefaultMesh"].visual.material.alphaMode == "MASK"
        assert rs.geometry["TestCutoff25Mesh"].visual.material.alphaMode == "MASK"
        assert rs.geometry["TestCutoff25Mesh"].visual.material.alphaCutoff == 0.25
        assert rs.geometry["TestCutoff75Mesh"].visual.material.alphaMode == "MASK"
        assert rs.geometry["TestCutoff75Mesh"].visual.material.alphaCutoff == 0.75
        assert rs.geometry["TestBlendMesh"].visual.material.alphaMode == "BLEND"
        # defaults OPAQUE
        assert rs.geometry["TestOpaqueMesh"].visual.material.alphaMode is None

    def test_units(self):
        # Trimesh will store units as a GLTF extra if they
        # are defined so check that.

        original = g.get_mesh("pins.glb")

        # export it as a a GLB file
        export = original.export(file_type="glb")
        validate_glb(export)

        kwargs = g.trimesh.exchange.gltf.load_glb(g.trimesh.util.wrap_as_stream(export))
        # roundtrip it
        reloaded = load_kwargs(kwargs)
        # make basic assertions
        g.scene_equal(original, reloaded)

        # make assertions on original and reloaded
        for scene in [original, reloaded]:
            # units should be stored as an extra
            assert scene.units == "mm"

            # make sure we have two unique geometries
            assert len(scene.geometry) == 2
            # that should have seven instances
            assert len(scene.graph.nodes_geometry) == 7

            # all meshes should be well constructed
            assert all(m.is_volume for m in scene.geometry.values())

            # check unit conversions for fun
            extents = scene.extents.copy()
            as_in = scene.convert_units("in")
            # should all be exactly mm -> in conversion factor
            assert g.np.allclose(extents / as_in.extents, 25.4, atol=0.001)

        m = g.get_mesh("testplate.glb")
        assert m.units == "meters"

    def test_basic(self):
        # split a multibody mesh into a scene
        scene = g.trimesh.scene.split_scene(g.get_mesh("cycloidal.ply"))
        # should be 117 geometries
        assert len(scene.geometry) >= 117

        # a dict with {file name: str}
        export = scene.export(file_type="gltf")
        # load from just resolver
        r = g.trimesh.load(file_obj=None, file_type="gltf", resolver=export)

        # will assert round trip is roughly equal
        g.scene_equal(r, scene)

        # try loading from a ZIP archive
        zipped = g.trimesh.util.compress(export)
        r = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(zipped), file_type="zip"
        )

        # try loading from a file name
        # will require a file path resolver
        with g.TemporaryDirectory() as d:
            for file_name, data in export.items():
                with open(g.os.path.join(d, file_name), "wb") as f:
                    f.write(data)
            # load from file path of header GLTF
            rd = g.trimesh.load(g.os.path.join(d, "model.gltf"))
            # will assert round trip is roughly equal
            g.scene_equal(rd, scene)

    def test_merge_buffers(self):
        # split a multibody mesh into a scene
        scene = g.trimesh.scene.split_scene(g.get_mesh("cycloidal.ply"))

        # export a gltf with the merge_buffers option set to true
        export = scene.export(file_type="gltf", merge_buffers=True)

        # We should end up with a single .bin and scene.gltf
        assert len(export.keys()) == 2

        # reload the export
        reloaded = load_kwargs(
            g.trimesh.exchange.gltf.load_gltf(
                file_obj=None, resolver=g.trimesh.visual.resolvers.ZipResolver(export)
            )
        )

        # check to make sure the geometry keys are the same
        assert set(reloaded.geometry.keys()) == set(scene.geometry.keys())

    def test_merge_primitives(self):
        # test to see if the `merge_primitives` logic is working
        a = g.get_mesh("CesiumMilkTruck.glb")
        assert len(a.geometry) == 4

        # should combine the multiple primitives into a single mesh
        b = g.get_mesh("CesiumMilkTruck.glb", merge_primitives=True)
        assert len(b.geometry) == 2

    def test_specular_glossiness(self):
        s = g.get_mesh("pyramid.zip")
        assert len(s.geometry) > 0
        assert "GLTF" in s.geometry

        mat = s.geometry["GLTF"].visual.material
        assert isinstance(mat, g.trimesh.visual.material.PBRMaterial)

        color = g.np.array(mat.baseColorTexture)[:, :, :3]
        assert color.shape[0] == 84 and color.shape[1] == 71

        # reference values generated with:
        # https://kcoley.github.io/glTF/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows-bjs/
        assert g.np.allclose(color[0, 0], [247, 223, 190], atol=1)
        assert g.np.allclose(color[30, 30], [247, 226, 196], atol=1)
        assert g.np.allclose(color[60, 10], [249, 231, 203], atol=1)
        color = mat.baseColorFactor
        assert color.dtype == g.np.uint8
        assert g.np.allclose(color, [255, 255, 255, 255])

        metallic_roughness = (
            g.np.array(mat.metallicRoughnessTexture, dtype=g.np.float32) / 255.0
        )
        assert metallic_roughness.shape[0] == 84 and metallic_roughness.shape[1] == 71

        # https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
        metallic = metallic_roughness[:, :, 2]
        roughness = metallic_roughness[:, :, 1]

        assert g.np.allclose(metallic[0, 0], 0.231, atol=0.004)
        assert g.np.allclose(metallic[30, 30], 0.207, atol=0.004)
        assert g.np.allclose(metallic[60, 10], 0.133, atol=0.004)

        assert g.np.allclose(roughness[0, 0], 0.898, atol=0.004)
        assert g.np.allclose(roughness[30, 30], 0.902, atol=0.004)
        assert g.np.allclose(roughness[60, 10], 0.898, atol=0.004)

        assert mat.metallicFactor == 1.0
        assert mat.roughnessFactor == 1.0
        assert all(mat.emissiveFactor == [0.0, 0.0, 0.0])

    def test_spec_gloss_factors_only(self):
        # test that we can load a GLTF with specular/glossiness material without textures
        s = g.get_mesh("pbr_cubes_emissive_spec_gloss.zip")

        assert all(
            isinstance(m.visual.material, g.trimesh.visual.material.PBRMaterial)
            for m in s.geometry.values()
        )

        spec_gloss_mat = s.geometry["Cube.005"].visual.material
        # this is a special case, because color is only coming from specular.
        # the diffuse value is black
        assert g.np.allclose(spec_gloss_mat.baseColorFactor, [254, 194, 85, 255], atol=1)
        assert g.np.allclose(spec_gloss_mat.metallicFactor, 1.0)
        assert g.np.allclose(spec_gloss_mat.roughnessFactor, 0.3)

    def test_write_dir(self):
        # try loading from a file name
        # will require a file path resolver
        original = g.get_mesh("fuze.obj")
        assert isinstance(original, g.trimesh.Trimesh)
        s = original.scene()
        with g.TemporaryDirectory() as d:
            path = g.os.path.join(d, "heyy.gltf")
            s.export(file_obj=path)
            r = g.trimesh.load(path)
            assert isinstance(r, g.trimesh.Scene)
            assert len(r.geometry) == 1
            m = next(iter(r.geometry.values()))
            assert g.np.isclose(original.area, m.area)

    def test_merge_primitives_materials(self):
        # test to see if the `merge_primitives` logic is working
        a = g.get_mesh("rgb_cube_with_primitives.gltf", merge_primitives=True)
        assert len(a.geometry["Cube"].visual.material) == 3
        # what the face materials should be
        truth = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        assert g.np.allclose(a.geometry["Cube"].visual.face_materials, truth)
        # make sure copying did the things correctly
        c = a.copy()
        assert g.np.allclose(c.geometry["Cube"].visual.face_materials, truth)

    def test_merge_primitives_skip_materials(self):
        # no primitive has a visual with `skip_materials`
        a = g.get_mesh(
            "rgb_cube_with_primitives.gltf", merge_primitives=True, skip_materials=True
        )
        assert len(a.geometry) == 1
        assert len(a.geometry["Cube"].faces) == 12
        assert not hasattr(a.geometry["Cube"].visual, "material")

    def _primitives_glb(self, primitives):
        # build a GLB with one mesh of the given primitives over shared
        # accessors: 0 vertices, 1 and 2 triangles, 3 and 4 lines
        import json
        import struct

        vertices = g.np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
            dtype=g.np.float32,
        )
        indices = g.np.array([0, 1, 2, 3, 4, 5, 0, 3, 1, 2], dtype=g.np.uint16)
        binary = vertices.tobytes() + indices.tobytes()
        header = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": primitives}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes},
                {
                    "buffer": 0,
                    "byteOffset": vertices.nbytes,
                    "byteLength": indices.nbytes,
                },
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 6,
                    "type": "VEC3",
                    "min": [0, 0, 0],
                    "max": [1, 1, 1],
                }
            ]
            + [
                {
                    "bufferView": 1,
                    "byteOffset": offset,
                    "componentType": 5123,
                    "count": count,
                    "type": "SCALAR",
                }
                for offset, count in ((0, 3), (6, 3), (12, 2), (16, 2))
            ],
        }
        content = json.dumps(header).encode()
        content += b" " * (-len(content) % 4)
        return (
            struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(content) + 8 + len(binary))
            + struct.pack("<II", len(content), 0x4E4F534A)
            + content
            + struct.pack("<II", len(binary), 0x004E4942)
            + binary
        )

    def test_merge_primitives_lines(self):
        # a mesh made of only GL_LINES primitives has no faces to merge
        line = {"attributes": {"POSITION": 0}, "mode": 1}
        glb = self._primitives_glb([dict(line, indices=3), dict(line, indices=4)])
        for merge in (False, True):
            s = g.trimesh.load_scene(
                g.trimesh.util.wrap_as_stream(glb),
                file_type="glb",
                merge_primitives=merge,
                skip_materials=True,
            )
            assert len(s.geometry) == 2

    def test_merge_primitives_triangles_and_lines(self):
        # triangles are merged while the lines primitive stays separate
        tri = {"attributes": {"POSITION": 0}}
        line = {"attributes": {"POSITION": 0}, "mode": 1}
        glb = self._primitives_glb(
            [dict(tri, indices=1), dict(tri, indices=2), dict(line, indices=3)]
        )
        for merge, count in ((True, 2), (False, 3)):
            s = g.trimesh.load_scene(
                g.trimesh.util.wrap_as_stream(glb),
                file_type="glb",
                merge_primitives=merge,
                skip_materials=True,
            )
            assert len(s.geometry) == count
            meshes = [m for m in s.geometry.values() if isinstance(m, g.trimesh.Trimesh)]
            assert len(meshes) == count - 1
            assert sum(len(m.faces) for m in meshes) == 2
            assert len(s.geometry) - len(meshes) == 1

    def test_merge_primitives_materials_roundtrip(self):
        # test to see if gltf loaded with `merge_primitives`
        # and then exported back
        # to gltf, produces a valid gltf.
        a = g.get_mesh("rgb_cube_with_primitives.gltf", merge_primitives=True)
        result = a.export(file_type="gltf", merge_buffers=True)
        with g.TemporaryDirectory() as d:
            for file_name, data in result.items():
                with open(g.os.path.join(d, file_name), "wb") as f:
                    f.write(data)

            rd = g.trimesh.load(g.os.path.join(d, "model.gltf"), merge_primitives=True)
            assert isinstance(rd, g.trimesh.Scene)
            # will assert round trip is roughly equal
            # TODO : restore
            # g.scene_equal(rd, a)

    def test_optional_camera(self):
        gltf_cameras_key = "cameras"

        # if there's no camera in the scene, then it shouldn't be added to the
        # gltf
        box = g.trimesh.creation.box([1, 1, 1])
        scene = g.trimesh.Scene(box)
        export = scene.export(file_type="gltf")
        assert gltf_cameras_key not in g.json.loads(export["model.gltf"].decode("utf8"))

        # `scene.camera` creates a camera if it does not exist.
        # once in the scene, it should be added to the gltf.
        box = g.trimesh.creation.box([1, 1, 1])
        scene = g.trimesh.Scene(box)
        scene.set_camera()
        export = scene.export(file_type="gltf")
        assert gltf_cameras_key in g.json.loads(export["model.gltf"].decode("utf8"))

    def test_gltf_pole(self):
        scene = g.get_mesh("simple_pole.glb")

        # should have multiple primitives
        assert len(scene.geometry) == 11

        export = scene.export(file_type="glb")
        validate_glb(export)
        # check a roundtrip
        reloaded = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")
        # make basic assertions
        g.scene_equal(scene, reloaded)

    def test_material_primary_colors(self):
        primary_color_material = g.trimesh.visual.material.PBRMaterial()
        primary_color_material.baseColorFactor = (255, 0, 0, 255)
        sphere = g.trimesh.creation.icosphere()
        sphere.visual = g.trimesh.visual.TextureVisuals(material=primary_color_material)
        sphere.visual.material = primary_color_material
        # material will *not* export without uv coordinates to gltf
        # as GLTF requires TEXCOORD_0 be defined if there is a material
        sphere.visual.uv = g.np.zeros((len(sphere.vertices), 2))
        scene = g.trimesh.Scene([sphere])

        def to_integer(args):
            args["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] = [1, 0, 0, 1]

        export = scene.export(file_type="glb", tree_postprocessor=to_integer)
        validate_glb(export)
        reloaded = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(export), file_type="glb"
        )
        assert len(reloaded.geometry) == 1
        # get meshes back
        sphere_b = next(iter(reloaded.geometry.values()))
        assert (sphere_b.visual.material.baseColorFactor == (255, 0, 0, 255)).all()

    def test_material_hash(self):
        # load mesh twice independently
        a = g.get_mesh("fuze.obj")
        b = g.get_mesh("fuze.obj")
        # move one of the meshes away from the other
        a.apply_translation([a.scale, 0, 0])

        # materials should not be the same object
        assert id(a.visual.material) != id(b.visual.material)
        # despite being loaded separately material hash should match
        assert hash(a.visual.material) == hash(b.visual.material)

        # create a scene with two meshes
        scene = g.trimesh.Scene([a, b])
        # get the exported GLTF header of a scene with both meshes
        header = g.json.loads(
            scene.export(file_type="gltf", unitize_normals=True)["model.gltf"].decode(
                "utf-8"
            )
        )
        # header should contain exactly one material
        assert len(header["materials"]) == 1
        # both meshes should be contained in the export
        assert len(header["meshes"]) == 2

        # get a reloaded version
        export = scene.export(file_type="glb", unitize_normals=True)
        validate_glb(export)
        reloaded = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(export), file_type="glb"
        )

        # meshes should have survived
        assert len(reloaded.geometry) == 2
        # get meshes back
        ar, br = reloaded.geometry.values()

        # should have been loaded as a PBR material
        assert isinstance(ar.visual.material, g.trimesh.visual.material.PBRMaterial)

        # materials should have the same memory location
        assert id(ar.visual.material) == id(br.visual.material)

        # make sure hash is returning something
        ahash = hash(ar.visual.material)
        # should be returning valid material hashes
        assert isinstance(ahash, int)
        assert ahash != 0

    def test_node_name(self):
        # Test to see if node names generally survive
        # an export-import cycle.

        # a scene
        s = g.get_mesh("cycloidal.3DXML")
        # export as GLB then re-load
        export = s.export(file_type="glb")
        validate_glb(export)
        r = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")
        # make sure we have the same geometries before and after
        assert set(s.geometry.keys()) == set(r.geometry.keys())
        # make sure the node names are the same before and after
        assert set(s.graph.nodes_geometry) == set(r.graph.nodes_geometry)

    def test_nested_scale(self):
        # nested transforms with scale
        s = g.get_mesh("nested.glb")
        assert len(s.graph.nodes_geometry) == 3
        assert g.np.allclose(
            [[-1.16701, -2.3366, -0.26938], [0.26938, 1.0, 0.26938]], s.bounds, atol=1e-4
        )

    def test_schema(self):
        # get a copy of the GLTF schema and do simple checks
        s = g.trimesh.exchange.gltf.get_schema()

        # make sure it has at least the keys we expect
        assert set(s["properties"].keys()).issuperset(
            {
                "accessors",
                "animations",
                "asset",
                "buffers",
                "bufferViews",
                "cameras",
                "images",
                "materials",
                "meshes",
                "nodes",
                "samplers",
                "scene",
                "scenes",
                "skins",
                "textures",
                "extensions",
                "extras",
            }
        )

        # lightly check to see that no references exist
        assert "$ref" not in g.json.dumps(s)

        # `gltf.validate` skips the meta-schema check so do it here once
        validator = g.jsonschema.validators.validator_for(s)
        assert validator(validator.META_SCHEMA).is_valid(s)

        # validation is cached so mutating a returned copy must not change it
        s["required"].append("missing")
        g.trimesh.exchange.gltf.validate({"asset": {"version": "2.0"}})

    def test_export_custom_attributes(self):
        # Write and read custom vertex attributes to gltf
        sphere = g.trimesh.primitives.Sphere()
        v_count, _ = sphere.vertices.shape

        random = g.np.random.default_rng(seed=0)
        sphere.vertex_attributes["_CustomFloat32Scalar"] = random.random(
            (v_count, 1)
        ).astype(g.np.float32)
        sphere.vertex_attributes["_CustomFloat32Vec3"] = random.random(
            (v_count, 3)
        ).astype(g.np.float32)
        sphere.vertex_attributes["_CustomFloat32Mat4"] = random.random(
            (v_count, 4, 4)
        ).astype(g.np.float32)

        # export as GLB bytes
        export = sphere.export(file_type="glb")
        # this should validate just fine
        validate_glb(export)

        # uint32 is slightly off-label and may cause
        # validators to fail but if you're a bad larry who
        # doesn't follow the rules it should be fine
        sphere.vertex_attributes["_CustomUInt32Scalar"] = random.integers(
            0, 1000, size=(v_count, 1)
        ).astype(g.np.uint32)

        # when you add a uint16/int16 the gltf-validator
        # complains about the 4-byte boundaries even though
        # all their lengths and offsets mod 4 are zero
        # not sure if that's a validator bug or what
        sphere.vertex_attributes["_CustomUInt16Scalar"] = random.integers(
            0, 1000, size=(v_count, 1)
        ).astype(g.np.uint16)
        sphere.vertex_attributes["_CustomInt16Scalar"] = random.integers(
            0, 1000, size=(v_count, 1)
        ).astype(g.np.int16)

        # export as GLB then re-load
        export = sphere.export(file_type="glb")

        r = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")

        for _, val in r.geometry.items():
            assert set(val.vertex_attributes.keys()) == set(
                sphere.vertex_attributes.keys()
            )
            for key in val.vertex_attributes:
                # the vertex attribute before round-tripping
                ori = sphere.vertex_attributes[key]

                # non 4-byte aligned attributes would have been padded
                check = val.vertex_attributes[key][:, : ori.shape[1]]

                assert g.np.allclose(ori, check), key

    def test_extras(self):
        # if GLTF extras are defined, make sure they survive a round trip
        s = g.get_mesh("cycloidal.3DXML")

        scene_extensions = {"EXT_trimesh_test": {"ext_data": 1.23}}
        # some dummy data
        dummy = {
            "who": "likes cheese",
            "potatoes": 25,
            "gtlf_extensions": scene_extensions,
        }

        # export as GLB with extras passed to the exporter then re-load
        s.metadata = dummy
        export = s.export(file_type="glb")
        validate_glb(export)
        r = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")

        # make sure extras survived a round trip
        assert all(r.metadata[k] == v for k, v in dummy.items())

    def test_extras_nodes(self):
        mesh_extensions = {"EXT_trimesh_test": {"ext_data": 1.23}}
        test_metadata = {
            "test_str": "test_value",
            "test_int": 1,
            "test_float": 0.123456789,
            "test_bool": True,
            "test_array": [1, 2, 3],
            "test_dict": {"a": 1, "b": 2},
            "gltf_extensions": mesh_extensions,
        }

        sphere1 = g.trimesh.primitives.Sphere(radius=1.0)
        sphere1.metadata.update(test_metadata)
        sphere2 = g.trimesh.primitives.Sphere(radius=2.0)
        sphere2.metadata.update(test_metadata)

        tf1 = g.trimesh.transformations.translation_matrix([0, 0, -2])
        tf2 = g.trimesh.transformations.translation_matrix([5, 5, 5])

        s = g.trimesh.scene.Scene()
        s.add_geometry(
            sphere1,
            node_name="Sphere1",
            geom_name="Geom Sphere1",
            transform=tf1,
            metadata={"field": "extra_data1"},
        )
        node_extensions = {"EXT_trimesh_test": {"ext_data": 1.23}}
        sphere2_metadata = {"field": "extra_data2", "gltf_extensions": node_extensions}
        s.add_geometry(
            sphere2,
            node_name="Sphere2",
            geom_name="Geom Sphere2",
            parent_node_name="Sphere1",
            transform=tf2,
            metadata=sphere2_metadata,
        )

        # Test extras appear in the exported model nodes
        files = s.export(None, "gltf")
        gltf_data = files["model.gltf"]
        assert "test_value" in gltf_data.decode("utf8")

        # Check node extras survive a round trip
        export = s.export(file_type="glb")
        validate_glb(export)
        r = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")
        files = r.export(None, "gltf")
        gltf_data = files["model.gltf"]
        # Check that the mesh and node metadata/extras survived
        assert "test_value" in gltf_data.decode("utf8")
        assert "extra_data1" in gltf_data.decode("utf8")
        # Check that the extensions were removed from the metadata;
        # they should be saved as 'extensions' in the gltf file
        assert "gltf_extensions" not in gltf_data.decode("utf8")

        # Check that the node transforms and metadata/extras survived
        edge = r.graph.transforms.edge_data[("world", "Sphere1")]
        assert g.np.allclose(edge["matrix"], tf1)
        assert edge["metadata"]["field"] == "extra_data1"

        edge = r.graph.transforms.edge_data[("Sphere1", "Sphere2")]
        assert g.np.allclose(edge["matrix"], tf2)
        assert edge["metadata"]["field"] == "extra_data2"
        # Check that the node's extensions survived
        assert edge["metadata"]["gltf_extensions"] == node_extensions

        # Check that the mesh extensions survived
        for mesh in r.geometry.values():
            assert mesh.metadata["gltf_extensions"] == mesh_extensions

        # all geometry should be the same
        assert set(r.geometry.keys()) == set(s.geometry.keys())
        for mesh in r.geometry.values():
            # metadata should have all survived
            assert all(mesh.metadata[k] == v for k, v in test_metadata.items())

    def test_read_scene_extras(self):
        # loads a glb with scene extras
        scene = g.get_mesh("monkey.glb", process=False)

        # expected data
        check = {"name": "monkey", "age": 32, "height": 0.987}

        meta = scene.metadata
        for key in check:
            # \check key existence and value
            assert key in meta
            assert meta[key] == check[key]

    def test_load_empty_nodes(self):
        # loads a glb with no meshes
        scene = g.get_mesh("empty_nodes.glb", process=False)

        # expected data
        check = {
            "parent": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "children_1": [
                [1.0, 0.0, 0.0, -5.0],
                [0.0, 1.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "children_2": [
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 1.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }

        # get the scene nodes
        objs = scene.graph.to_flattened()

        # check number
        assert len(objs) == 3

        for key in check:
            assert key in objs
            assert objs[key]["transform"] == check[key]

    def test_same_name(self):
        s = g.get_mesh("TestScene.gltf")
        # hardcode correct bounds to check against
        bounds = s.to_mesh().bounds

        # icosahedrons have two primitives each
        g.log.debug(len(s.geometry), len(s.graph.nodes_geometry))
        assert len(s.graph.nodes_geometry) == 9
        assert len(s.geometry) == 7
        assert g.np.allclose(s.bounds, bounds, atol=1e-3)

        # if merged should have combined the icosahedrons
        s = g.get_mesh("TestScene.gltf", merge_primitives=True)
        assert len(s.graph.nodes_geometry) == 7
        assert len(s.geometry) == 6
        assert g.np.allclose(s.bounds, bounds, atol=1e-3)

    def test_vertex_colors(self):
        # get a mesh with face colors
        m = g.get_mesh("machinist.XAML")
        # export as GLB then re-import
        export = m.export(file_type="glb")
        validate_glb(export)
        r = next(
            iter(
                g.trimesh.load(
                    g.trimesh.util.wrap_as_stream(export), file_type="glb"
                ).geometry.values()
            )
        )
        # original mesh should have vertex colors
        assert m.visual.kind == "face"
        assert g.np.ptp(g.np.ptp(m.visual.vertex_colors, axis=0)) > 0
        # vertex colors should have survived import-export
        assert g.np.allclose(m.visual.vertex_colors, r.visual.vertex_colors)

    def test_vertex_attrib(self):
        # test concatenation with texture
        m = g.get_mesh("fuze.obj")

        colors = (g.random((len(m.vertices), 4)) * 255).astype(g.np.uint8)

        # set the color vertex attribute
        m.visual.vertex_attributes["color"] = colors
        export = m.export(file_type="glb", unitize_normals=True)
        validate_glb(export)
        r = next(
            iter(
                g.trimesh.load(
                    g.trimesh.util.wrap_as_stream(export), file_type="glb"
                ).geometry.values()
            )
        )

        # make sure the color vertex attributes survived the roundtrip
        assert g.np.allclose(r.visual.vertex_attributes["color"], colors)

    def test_vertex_colors_import(self):
        # get a mesh with face colors
        m = g.get_mesh("cubevc.glb")
        assert len(m.geometry.items()) > 0

        mesh = next(iter(m.geometry.items()))[1]
        assert mesh is not None

        # Loaded mesh should have vertex colors
        assert hasattr(mesh.visual, "vertex_colors")

        # Loaded mesh should have all vertex colors filled with magenta color
        magenta = g.np.array([255, 0, 255, 255], dtype=g.np.uint8)
        for color in mesh.visual.vertex_colors:
            is_magenta = g.np.array_equal(color, magenta)
            assert is_magenta, (
                f"Imported vertex color is not of expected value: got {color}, expected {magenta}"
            )

    def test_export_postprocess(self):
        scene = g.trimesh.Scene()
        sphere = g.trimesh.primitives.Sphere()
        sphere.visual.material = g.trimesh.visual.material.PBRMaterial(name="unlit_test")
        scene.add_geometry(sphere)

        def add_unlit(gltf_tree):
            for material_dict in gltf_tree["materials"]:
                if "unlit" in material_dict.get("name", "").lower():
                    material_dict["extensions"] = {"KHR_materials_unlit": {}}
            gltf_tree["extensionsUsed"] = ["KHR_materials_unlit"]

        gltf_1 = g.trimesh.exchange.gltf.export_gltf(scene)
        gltf_2 = g.trimesh.exchange.gltf.export_gltf(scene, tree_postprocessor=add_unlit)

        def extract_materials(gltf_files):
            return g.json.loads(gltf_files["model.gltf"].decode("utf8"))["materials"]

        assert "extensions" not in extract_materials(gltf_1)[-1]
        assert "extensions" in extract_materials(gltf_2)[-1]

    def test_primitive_geometry_meta(self):
        # Model with primitives
        s = g.get_mesh("CesiumMilkTruck.glb")
        # check to see if names are somewhat sane
        assert set(s.geometry.keys()) == {
            "Cesium_Milk_Truck",
            "Cesium_Milk_Truck_1",
            "Cesium_Milk_Truck_2",
            "Wheels",
        }
        # Assert that primitive geometries are marked as such
        assert s.geometry["Cesium_Milk_Truck"].metadata["from_gltf_primitive"]
        assert s.geometry["Cesium_Milk_Truck_1"].metadata["from_gltf_primitive"]
        assert s.geometry["Cesium_Milk_Truck_2"].metadata["from_gltf_primitive"]
        # Assert that geometries that are not primitives
        # are not marked as such
        assert not s.geometry["Wheels"].metadata["from_gltf_primitive"]

        # make sure the flags survive being merged
        m = g.get_mesh("CesiumMilkTruck.glb", merge_primitives=True)
        # names should be non-insane
        assert set(m.geometry.keys()) == {"Cesium_Milk_Truck", "Wheels"}
        assert not s.geometry["Wheels"].metadata["from_gltf_primitive"]
        assert s.geometry["Cesium_Milk_Truck"].metadata["from_gltf_primitive"]

    def test_points(self):
        # test a simple pointcloud export-import cycle
        points = g.np.arange(30).reshape((-1, 3))

        # get a pointcloud object
        cloud = g.trimesh.PointCloud(points)

        # export as gltf
        export = g.trimesh.Scene(cloud).export(file_type="glb")
        validate_glb(export)
        reloaded = next(
            iter(
                g.trimesh.load_scene(
                    g.trimesh.util.wrap_as_stream(export), file_type="glb"
                ).geometry.values()
            )
        )
        # make sure points survived export and reload
        assert g.np.allclose(reloaded.vertices, points)

        # now try adding color
        colors = g.trimesh.visual.color.random_color(count=len(points))
        cloud.colors = colors
        export = g.trimesh.Scene(cloud).export(file_type="glb")
        validate_glb(export)
        reloaded = next(
            iter(
                g.trimesh.load_scene(
                    g.trimesh.util.wrap_as_stream(export), file_type="glb"
                ).geometry.values()
            )
        )

        # make sure points with color survived export and reload
        assert g.np.allclose(reloaded.vertices, points)
        assert g.np.allclose(reloaded.colors, colors)

    def test_world_node_collision(self):
        # a non-root node named "world" must not merge with the hardcoded base
        # frame; on main its transform collapses to identity on round-trip
        tf = g.trimesh.transformations.rotation_matrix(0.7, [1.0, 2.0, 3.0])
        tf[:3, 3] = [5.0, -3.0, 2.0]

        scene = g.trimesh.Scene(base_frame="root")
        scene.add_geometry(
            g.trimesh.creation.box(),
            node_name="world",
            parent_node_name="root",
            geom_name="box",
            transform=tf,
        )

        def geom_world_transform(s):
            return s.graph.get(s.graph.nodes_geometry[0])[0]

        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(scene.export(file_type="glb")),
            file_type="glb",
        )
        assert g.np.allclose(geom_world_transform(scene), geom_world_transform(reloaded))

    def test_world_root_transform(self):
        # a root node named "world" with a transform must keep it —
        # merging with the synthetic base frame silently dropped it
        scene = g.trimesh.load(world_root_tree(), file_type="gltf")
        transform = scene.graph.get(scene.graph.nodes_geometry[0])[0]
        assert g.np.allclose(transform[:3, 3], [10.0, 0.0, 0.0])

    def test_world_no_accumulation(self):
        # the 2025-08-11 unconditional-rename attempt failed because every
        # round-trip of trimesh's own exports renamed the wrapper node and
        # grew the graph — pin names and node count across cycles
        scene = g.trimesh.load(world_root_tree(), file_type="gltf")

        def cycle(s):
            return g.trimesh.load(
                g.trimesh.util.wrap_as_stream(s.export(file_type="glb")),
                file_type="glb",
            )

        once = cycle(scene)
        twice = cycle(once)
        thrice = cycle(twice)
        assert set(once.graph.nodes) == set(twice.graph.nodes)
        assert set(twice.graph.nodes) == set(thrice.graph.nodes)
        # the world transform of the geometry must be stable too
        for s in (once, twice, thrice):
            transform = s.graph.get(s.graph.nodes_geometry[0])[0]
            assert g.np.allclose(transform[:3, 3], [10.0, 0.0, 0.0])

    def test_export_no_wrapper_node(self):
        # exports hang children as real scene roots instead of wrapping
        # everything in a synthetic transform-less "world" node
        scene = g.trimesh.Scene(g.trimesh.creation.box())
        scene.metadata["hi"] = 3

        export = scene.export(file_type="gltf")
        tree = g.json.loads(export["model.gltf"].decode("utf-8"))

        # no wrapper node and the real nodes are the scene roots
        names = [n.get("name") for n in tree["nodes"]]
        assert scene.graph.base_frame not in names
        assert len(tree["nodes"]) == len(scene.graph.nodes) - 1
        assert len(tree["scenes"][0]["nodes"]) == len(
            scene.graph.transforms.children[scene.graph.base_frame]
        )
        # scene metadata must survive the root change
        assert tree["scenes"][0]["extras"]["hi"] == 3

        # round-trips keep the node names identical
        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(scene.export(file_type="glb")),
            file_type="glb",
        )
        assert set(reloaded.graph.nodes) == set(scene.graph.nodes)

    def test_bulk(self):
        # Try exporting every loadable model to GLTF and checking
        # the generated header against the schema.

        # strict mode runs a schema header validation
        assert g.trimesh.tol.strict

        # check mesh, path, pointcloud exports
        for root in [g.dir_models, g.os.path.join(g.dir_models, "2D")]:
            for fn in g.os.listdir(root):
                path_in = g.os.path.join(root, fn)
                try:
                    geom = g.trimesh.load(path_in)
                    if isinstance(geom, g.trimesh.path.path.Path):
                        geom = g.trimesh.Scene(geom)
                except BaseException as E:
                    g.log.debug(E)
                    continue
                # voxels don't have an export to gltf mode
                if isinstance(geom, g.trimesh.voxel.VoxelGrid):
                    try:
                        geom.export(file_type="glb")
                    except ValueError:
                        # should have raised so all good
                        continue
                    raise ValueError("voxel was allowed to export wrong GLB!")
                if hasattr(geom, "vertices") and len(geom.vertices) == 0:
                    continue
                if hasattr(geom, "geometry") and len(geom.geometry) == 0:
                    continue

                g.log.info(f"Testing: {fn}")
                # check a roundtrip which will validate on export
                # and crash on reload if we've done anything screwey
                # unitize normals will unitize any normals to comply with
                # the validator although there are probably reasons you'd
                # want to roundtrip non-unit normals for things, stuff, and
                # activities
                export = geom.export(file_type="glb", unitize_normals=True)
                validate_glb(export, name=fn)

                # shouldn't crash on a reload
                reloaded = g.trimesh.load(
                    file_obj=g.trimesh.util.wrap_as_stream(export), file_type="glb"
                )

                if isinstance(geom, g.trimesh.Trimesh):
                    assert g.np.isclose(geom.area, reloaded.area)

                # compute some stuff
                assert isinstance(reloaded.area, float)
                assert isinstance(reloaded.duplicate_nodes, list)

    def test_interleaved(self):
        # do a quick check on a mesh that uses byte stride
        with open(g.get_path("BoxInterleaved.glb"), "rb") as f:
            k = g.trimesh.exchange.gltf.load_glb(f)
        # get the kwargs for the mesh constructor
        c = k["geometry"]["Mesh"]
        # should have vertex normals
        assert c["vertex_normals"].shape == c["vertices"].shape
        # interleaved vertex normals should all be unit vectors
        assert g.np.allclose(1.0, g.np.linalg.norm(c["vertex_normals"], axis=1))

        # should also load as a box
        m = g.get_mesh("BoxInterleaved.glb").geometry["Mesh"]
        assert g.np.isclose(m.volume, 1.0)

    def test_equal_by_default(self):
        # all things being equal we shouldn't be moving things
        # for the usual load-export loop
        s = g.get_mesh("fuze.obj")
        # export as GLB then re-load
        export = s.export(file_type="glb", unitize_normals=True)
        validate_glb(export)
        reloaded = g.trimesh.load(
            g.trimesh.util.wrap_as_stream(export), file_type="glb", process=False
        )
        assert len(reloaded.geometry) == 1
        m = next(iter(reloaded.geometry.values()))
        assert g.np.allclose(m.visual.uv, s.visual.uv)
        assert g.np.allclose(m.vertices, s.vertices)
        assert g.np.allclose(m.faces, s.faces)

        # will run a kdtree check
        g.texture_equal(s, m)

    def test_gltf_by_name(self):
        m = g.trimesh.creation.icosphere()

        with g.TemporaryDirectory() as d:
            # export the GLTF file by name
            file_path = g.os.path.join(d, "hi.gltf")
            # export the file by path
            m.export(file_path)
            # reload the gltf from the file path
            r = g.trimesh.load(file_path)

            assert isinstance(r, g.trimesh.Scene)
            assert len(r.geometry) == 1
            assert g.np.isclose(next(iter(r.geometry.values())).volume, m.volume)

    def test_embed_buffer(self):
        scene = g.trimesh.Scene(
            {
                "thing": g.trimesh.primitives.Sphere(),
                "other": g.trimesh.creation.capsule(),
            }
        )

        with g.TemporaryDirectory() as D:
            path = g.os.path.join(D, "hi.gltf")
            scene.export(path)

            # should export with separate buffers
            assert len(g.os.listdir(D)) > 1

            reloaded = g.trimesh.load(path)
            assert set(reloaded.geometry.keys()) == set(scene.geometry.keys())

        with g.TemporaryDirectory() as D:
            path = g.os.path.join(D, "hi.gltf")
            scene.export(path, embed_buffers=True)

            # should export with embedded buffers
            assert len(g.os.listdir(D)) == 1

            reloaded = g.trimesh.load(path)
            assert set(reloaded.geometry.keys()) == set(scene.geometry.keys())

    def test_webp(self):
        # load textured file
        mesh = g.get_mesh("fuze.ply")
        assert hasattr(mesh.visual, "uv")

        for extension in ["glb"]:
            export = mesh.export(file_type=extension, extension_webp=True)
            validate_glb(export)

            # roundtrip
            reloaded = g.trimesh.load(
                g.trimesh.util.wrap_as_stream(export), file_type=extension
            )

            g.scene_equal(g.trimesh.Scene(mesh), reloaded)

    def test_relative_paths(self):
        # try with a relative path
        cwd = g.os.path.abspath(g.os.path.expanduser("."))
        with g.TemporaryDirectory() as d:
            g.os.makedirs(g.os.path.join(d, "fused"))
            g.os.chdir(d)
            g.trimesh.creation.box().export("fused/hi.gltf")
            r = g.trimesh.load("fused/hi.gltf")
            assert g.np.isclose(r.volume, 1.0)

            # avoid a windows file-access error
            g.os.chdir(cwd)

        with g.TemporaryDirectory() as d:
            # now try it without changing to that directory
            full = g.os.path.join(d, "hi", "there", "different", "levels")
            path = g.os.path.join(full, "hey.gltf")
            g.os.makedirs(full)
            g.trimesh.creation.box().export(path)
            r = g.trimesh.load(path)
            assert g.np.isclose(r.volume, 1.0)

    def test_postprocess(self):
        # check to see if keys we expect exist
        s = g.get_mesh("cycloidal.3DXML")

        def post(tree):
            # should have exported meshes here
            assert len(tree["meshes"]) == len(s.geometry)
            # should have buffers
            assert len(tree["buffers"]) >= 1

        # export with a postprocessor
        s.export(file_type="glb", tree_postprocessor=post)

    def test_unitize_normals_null_values(self):
        # Create the mesh
        mesh = g.trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]],
            faces=[[0, 1, 2], [1, 3, 2], [0, 1, 4]],
        )

        # Set the normal of the first vertex to null
        modified_normals = mesh.vertex_normals.copy()
        modified_normals[0] = [0, 0, 0]

        mesh.vertex_normals = modified_normals

        # Export the mesh
        export = mesh.export(file_type="glb", unitize_normals=True)
        reimported_mesh = next(
            iter(
                g.trimesh.load(
                    g.trimesh.util.wrap_as_stream(export), file_type="glb"
                ).geometry.values()
            )
        )

        # Check that the normals are still null
        assert g.np.allclose(reimported_mesh.vertex_normals[0], [0, 0, 0])

    def test_no_indices(self):
        # test mesh with no indices (faces should be generated correctly)
        mesh = g.get_mesh("no_indices_3storybuilding.glb")
        assert len(mesh.triangles) == 72

        # the mesh is actually mode 5 with 4 vertices
        # which as triangle strips would be 2 faces
        mesh = g.get_mesh("Mesh_PrimitiveMode_04.gltf")
        assert len(mesh.triangles) == 2

    def test_simple_material_conversion(self):
        # make sure the name is preserved for a material
        mesh = next(iter(g.get_mesh("Duck.glb").geometry.values()))

        mat = mesh.visual.material
        assert isinstance(mat, g.trimesh.visual.material.PBRMaterial)
        assert isinstance(mat.name, str)
        assert len(mat.name) > 0

        simple = mesh.visual.material.to_simple()
        assert simple.name == mesh.visual.material.name

        # the factor is the textbook sRGB decoding of the diffuse, stored
        # in 8 bits so within half a level of it, alpha is never decoded
        levels = g.np.arange(256, dtype=g.np.uint8)
        rgba = g.np.column_stack([levels, levels[::-1], levels // 2, levels])
        srgb = rgba[:, :3] / 255.0
        exact = g.np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        to_pbr = [
            g.trimesh.visual.material.SimpleMaterial(diffuse=c).to_pbr() for c in rgba
        ]
        factor = g.np.array([pbr.baseColorFactor for pbr in to_pbr])
        assert (g.np.abs(factor[:, :3] / 255.0 - exact) <= 0.5 / 255.0 + 1e-9).all()
        assert g.np.array_equal(factor[:, 3], rgba[:, 3]), "alpha untouched"

    def test_webp_roundtrip(self):
        m = g.get_mesh("fuze.obj")
        e = m.export(file_type="glb", extension_webp=True)
        r = g.trimesh.load_mesh(g.trimesh.util.wrap_as_stream(e), file_type="glb")

        # compare RGBA images for the roundtripped texture
        # make sure the webp roundtrip wasn't crazy
        a = g.np.array(m.visual.material.image)
        b = g.np.array(r.visual.material.baseColorTexture)

        assert a.shape == b.shape

        # roundtrip with a codec produces artifacts
        # if they are much different this number will be absolutely huge
        mean_squared_error = ((a - b) ** 2).sum() / g.np.prod(a.shape)
        assert mean_squared_error < 10.0


def test_scene_table():
    """
    Every built scene is a valid fixed point of loading which keeps its
    nodes, poses, lights and materials and writes no unused accessor, and
    edits of the files load back with the spec's defaults.
    """
    from PIL import Image

    from trimesh.scene.animation import Animation

    tf, lighting = g.trimesh.transformations, g.trimesh.scene.lighting
    material = g.trimesh.visual.material.PBRMaterial
    rng = g.np.random.default_rng(3)
    box = g.trimesh.creation.box

    def capture(scene, edit=None):
        """A scene loaded back from its GLB and the glTF tree written for it."""
        trees = []
        edit = edit or (lambda tree: None)
        exported = scene.export(
            file_type="glb", tree_postprocessor=lambda t: (edit(t), trees.append(t))
        )
        return g.roundtrip(exported, "glb"), trees[0]

    def described(scene):
        """Every light as its type, values and world transform."""
        if not scene.has_lights:
            # asking for `Scene.lights` would make the default ones
            return []
        return [
            (type(light).__name__, light.name, light.color.tolist(), light.intensity)
            + (light.radius, getattr(light, "outerConeAngle", None))
            + (scene.graph[light.name][0].round(6).tolist(),)
            for light in scene.lights
        ]

    # a cubic node under a parent which carries a `matrix` and steps in
    # scale, as two animations sharing a name are one glTF animation
    rig = g.trimesh.Scene()
    rig.add_geometry(box(), node_name="mover", parent_node_name="rig")
    rig.add_geometry(box(), node_name="still", transform=tf.translation_matrix([5, 0, 0]))
    rest = tf.rotation_matrix(0.6, [1, 1, 0], [2, 0, 0])
    rig.graph.update(frame_to="rig", matrix=rest)
    times = g.np.linspace(0.0, 2.0, 9)
    mover = tf.rotation_matrix(times * g.np.pi, [0, 0, 1])
    mover[:, :3, :3] *= rng.uniform(0.5, 2.0, (9, 1, 3))
    mover[:, :3, 3] = rng.normal(size=(9, 3))
    stretch = g.np.tile(g.np.eye(4), (5, 1, 1))
    stretch[:, [0, 1, 2], [0, 1, 2]] = rng.uniform(0.5, 2.0, (5, 3))
    rig.animations.extend(
        [
            Animation("spin", ["mover"], times, mover, interpolation="cubic"),
            Animation("spin", ["rig"], times[::2], rest @ stretch, interpolation="step"),
        ]
    )
    # punctual lights, and a lossless texture shared by two materials
    lit = g.trimesh.Scene(box())
    lit.lights = [
        lighting.DirectionalLight(name="key", color=[255, 244, 224, 255]),
        lighting.PointLight(name="fill", intensity=7.5, radius=12.0),
        lighting.SpotLight(name="rim", innerConeAngle=0.2, outerConeAngle=0.6),
    ]
    for light, x in zip(lit.lights, [1.0, -2.0, 3.0]):
        lit.graph[light.name] = tf.translation_matrix([x, 0, 2])

    def image(shape):
        """A random image of a shape, alpha and all."""
        return Image.fromarray(rng.integers(0, 255, shape, g.np.uint8))

    shared = image((8, 8, 3))
    # every field of each extension from the specs, not trimesh's table
    fields = {
        "KHR_materials_transmission": {
            "transmissionFactor": 0.8,
            "transmissionTexture": image((4, 6, 3)),
        },
        "KHR_materials_diffuse_transmission": {
            "diffuseTransmissionFactor": 0.3,
            "diffuseTransmissionTexture": image((4, 6, 4)),
            "diffuseTransmissionColorFactor": [0.2, 0.9, 0.1],
            "diffuseTransmissionColorTexture": image((5, 3, 3)),
        },
        "KHR_materials_unlit": {"unlit": True},
        "shared": {"baseColorTexture": shared, "occlusionTexture": shared},
        "reused": {"baseColorTexture": shared, "roughnessFactor": 0.5},
    }
    textured = g.trimesh.Scene()
    for name, values in fields.items():
        mesh = box()
        mesh.visual = g.trimesh.visual.TextureVisuals(
            uv=rng.random((len(mesh.vertices), 2)), material=material(**values)
        )
        textured.add_geometry(mesh, geom_name=name)
    # a mesh without UVs can't look a texture up so stores only scalars
    bare = box()
    bare.visual = g.trimesh.visual.TextureVisuals(
        material=material(baseColorTexture=shared, metallicFactor=0.25)
    )
    textured.add_geometry(bare, geom_name="bare")
    ortho = g.dolly_scene("box")
    ortho.camera = g.trimesh.scene.Camera(name="lens", magnification=[2.0, 1.5])
    sources = {
        "rig": rig,
        "box": g.dolly_scene("box"),
        "dolly": g.dolly_scene("dolly"),
        "ortho": ortho,
        "lit": lit,
        "textured": textured,
        "truck": g.get_mesh("CesiumMilkTruck.glb"),
    }
    for name, source in sources.items():
        g.assert_glb_roundtrip(source, name)
        loaded, tree = capture(source)
        assert described(loaded) == described(source), f"{name} lights"
        used = {
            sampler[k]
            for a in tree.get("animations", [])
            for sampler in a["samplers"]
            for k in ("input", "output")
        }
        used.update(s["inverseBindMatrices"] for s in tree.get("skins", []))
        for primitive in (p for m in tree["meshes"] for p in m["primitives"]):
            used.update((*primitive["attributes"].values(), primitive.get("indices")))
        assert used - {None} == set(range(len(tree["accessors"]))), f"{name} accessors"
    # a scene which never had them doesn't grow animations or lights
    plain = capture(g.trimesh.Scene(box()))[1]
    assert "animations" not in plain and "KHR_lights" not in str(plain)

    loaded, tree = capture(textured)
    for name, values in fields.items():
        stored = tree["materials"][
            tree["meshes"][list(textured.geometry).index(name)]["primitives"][0][
                "material"
            ]
        ]
        if name.startswith("KHR"):
            assert list(stored["extensions"]) == [name], f"{name} only"
        for key, value in values.items():
            # JSON stores floats exactly, and images are lossless PNG
            copy = getattr(loaded.geometry[name].visual.material, key)
            assert g.np.array_equal(g.np.asarray(copy), g.np.asarray(value)), key
    assert len(tree["images"]) == 4, "each image stored once"
    assert "baseColorTexture" not in tree["materials"][-1]["pbrMetallicRoughness"]
    assert bare.visual.material.baseColorTexture is shared, "no uv source untouched"

    # a node rotating in STEP beside a LINEAR slide is one glTF animation
    # whose steps hold between keyframes on import
    keys = g.np.linspace(0.0, 2.0, 5)
    turn = tf.rotation_matrix(keys, [0, 0, 1])[:, None]
    slide = g.np.tile(tf.translation_matrix([0, 0, 0]), (5, 1, 1, 1))
    slide[:, 0, :3, 3] = rng.normal(size=(5, 3))
    walk = g.dolly_scene("box")
    walk.animations = [
        Animation("walk", ["box"], keys, turn, interpolation="step"),
        Animation("walk", ["box"], keys, slide),
    ]
    loaded = capture(walk)[0]
    for time in g.np.linspace(-0.5, 2.5, 31):
        expected = walk.animations[1].interpolate(time) @ walk.animations[0].interpolate(
            time
        )
        loaded.animate(time)
        assert g.np.allclose(loaded.graph["box"][0], expected[0], atol=1e-6), time

    def strip(tree):
        """Drop optional fields: a light's color, a camera's aspect and far."""
        tree["extensions"]["KHR_lights_punctual"]["lights"][0].pop("color")
        for camera in tree["cameras"]:
            camera["perspective"].pop("aspectRatio")
            camera["perspective"].pop("zfar")

    stripped = g.dolly_scene("box")
    stripped.lights = lit.lights
    stripped.graph.update(frame_to="key")
    loaded = capture(stripped, strip)[0]
    rewritten = capture(loaded)[1]["cameras"]
    assert (loaded.lights[0].color == 255).all(), "a light without color is white"
    assert g.np.isclose(loaded.camera.fov[1], 45.0), "vertical fov"
    assert loaded.camera.z_far == g.np.inf, "missing zfar is infinite"
    assert all(len(c["perspective"]) == 2 for c in rewritten), "nothing invented"

    def elsewhere(tree):
        """Put a light on the node of a second scene, which isn't loaded."""
        name = "KHR_lights_punctual"
        tree["nodes"].append({"name": "elsewhere", "extensions": {name: {"light": 0}}})
        tree["scenes"].append({"nodes": [len(tree["nodes"]) - 1]})
        tree["extensions"] = {name: {"lights": [{"type": "point"}]}}
        tree["extensionsUsed"] = [name]

    assert not capture(g.trimesh.Scene(box()), elsewhere)[0].has_lights

    # joints without weights can't skin so they stay a custom attribute,
    # and joints and weights which aren't (n, 4) can't be a VEC4
    mesh = box()
    joints = g.np.ones((len(mesh.vertices), 4), dtype=g.np.uint16)
    mesh.vertex_attributes["joints"] = joints
    kept = capture(g.trimesh.Scene(mesh))[0].geometry.popitem()[1]
    assert g.np.array_equal(kept.vertex_attributes["_joints"], joints)
    mesh.vertex_attributes["weights"] = g.np.ones((len(mesh.vertices), 3))
    with g.pytest.raises(ValueError):
        mesh.export(file_type="glb")


@g.pytest.mark.parametrize(
    "values",
    [
        g.np.arange(8),
        g.np.arange(8, dtype=g.np.int32) - 3,
        g.np.arange(8) % 2 == 0,
        g.np.arange(8, dtype=">f8") / 4,
    ],
)
def test_export_custom_attribute_dtypes(values):
    # dtypes GLTF can't store directly should be cast rather than fail
    mesh = g.trimesh.creation.box()
    mesh.vertex_attributes["_values"] = values
    export = mesh.export(file_type="glb")
    validate_glb(export)
    scene = g.trimesh.load(g.trimesh.util.wrap_as_stream(export), file_type="glb")
    loaded = next(iter(scene.geometry.values())).vertex_attributes["_values"]
    # scalar attributes are padded out to 4 byte rows
    assert g.np.array_equal(loaded[:, 0], values)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
