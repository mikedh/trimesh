try:
    from . import generic as g
except BaseException:
    import generic as g

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
        g.validate_glb(export, "Duck.glb")

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

        # an image used by two materials and twice by one, and a twin of it
        # with a color profile glTF viewers ignore, are each stored once
        from PIL import Image, ImageCms

        image = Image.new("RGB", (64, 64), (200, 40, 40))
        profiled = image.copy()
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        profiled.info["icc_profile"] = profile
        for name, textures in {
            "shared": [image, image],
            "twins": [image, profiled],
        }.items():
            scene = g.trimesh.Scene()
            for index, texture in enumerate(textures):
                box = g.trimesh.creation.box()
                box.visual = g.trimesh.visual.TextureVisuals(
                    uv=box.vertices[:, :2] + 0.5,
                    material=g.trimesh.visual.material.PBRMaterial(
                        baseColorTexture=texture,
                        occlusionTexture=[texture, None][index],
                        roughnessFactor=0.25 * (index + 1),
                    ),
                )
                scene.add_geometry(
                    box, f"{name}_{index}", transform=tm([index * 2, 0, 0])
                )
            export = scene.export(file_type="glb")
            # the validator flags an image with a color profile
            g.validate_glb(export)
            header = g.glb_chunks(export)[0]
            counts = [len(header[key]) for key in ("images", "textures", "materials")]
            assert counts == [1, 1, 2], f"{name} stored once"
            for geometry in g.roundtrip(export, "glb").geometry.values():
                texture = geometry.visual.material.baseColorTexture
                assert (
                    "icc_profile" not in texture.info
                    and not hasattr(texture, "encoded")
                    and g.np.array_equal(g.np.asarray(texture), g.np.asarray(image))
                ), f"{name} pixels survive without their profile"
        # a mesh without UVs can't look a texture up so stores only the
        # scalars of its material and leaves the source material alone
        box = g.trimesh.creation.box()
        material = g.trimesh.visual.material.PBRMaterial(
            baseColorTexture=image, roughnessFactor=0.5, metallicFactor=0.25
        )
        box.visual = g.trimesh.visual.TextureVisuals(material=material)
        header = g.json.loads(box.export(file_type="gltf")["model.gltf"])
        stored = header["materials"][0]["pbrMetallicRoughness"]
        assert "images" not in header, "no uv stores no image"
        assert stored["roughnessFactor"] == 0.5 and stored["metallicFactor"] == 0.25, (
            "no uv scalars kept"
        )
        assert box.visual.material is material and material.baseColorTexture is image, (
            "no uv source untouched"
        )

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
        g.validate_glb(export)
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
        g.validate_glb(export, "fuze.ply")
        # make sure it works on meshes
        export = mesh.export(file_type="glb", unitize_normals=True)
        g.validate_glb(export, "fuze.ply")

        def images(data):
            """The bytes of every image stored in a GLB."""
            header, blob = g.glb_chunks(data)
            views = [header["bufferViews"][i["bufferView"]] for i in header["images"]]
            return [blob[v.get("byteOffset", 0) :][: v["byteLength"]] for v in views]

        def texture(scene):
            """The base color texture of the first geometry in a scene."""
            return next(iter(scene.geometry.values())).visual.material.baseColorTexture

        # textures loaded from a file are stored again exactly as they were
        # unless they changed, which is fast and doesn't grow the file, once
        # the color profile glTF doesn't allow in this one is dropped
        with open(g.os.path.join(g.dir_models, "BoxTextured.glb"), "rb") as f:
            source = g.from_glb_bytes(f.read()).export(file_type="glb")
        g.validate_glb(source, "BoxTextured.glb")
        loaded = g.from_glb_bytes(source)
        assert images(loaded.export(file_type="glb")) == images(source), "unchanged image"
        texture(loaded).paste((255, 0, 0), (0, 0, 8, 8))
        changed = texture(g.from_glb_bytes(loaded.export(file_type="glb")))
        assert g.np.array_equal(
            g.np.asarray(changed.convert("RGB")),
            g.np.asarray(texture(loaded).convert("RGB")),
        ), "changed image"

    def test_cesium(self):
        # A GLTF with a multi- primitive mesh

        s = g.get_mesh("CesiumMilkTruck.glb")
        # should be one Trimesh object per GLTF "primitive"
        assert len(s.geometry) == 4
        # every geometry displayed once, except wheels twice
        assert len(s.graph.nodes_geometry) == 5

        # make sure export doesn't crash
        export = s.export(file_type="glb")
        g.validate_glb(export)

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
        # the file is normal mapped without tangents, which viewers make
        # themselves as the spec says, so re-exporting adds none
        attributes = {
            name
            for mesh in g.glb_chunks(export)[0]["meshes"]
            for primitive in mesh["primitives"]
            for name in primitive["attributes"]
        }
        assert "TANGENT" not in attributes, attributes
        g.validate_glb(export)

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

        # every field of every material extension trimesh stores, from the
        # specs rather than trimesh's table: clear glass, a leaf scattering
        # light out of its other side, and a sky no light shades
        from PIL import Image

        random = g.np.random.default_rng(7)

        def image(shape):
            """A random image of a shape, alpha and all."""
            return Image.fromarray(random.integers(0, 255, shape, g.np.uint8))

        spec = {
            "glass": (
                "KHR_materials_transmission",
                {"transmissionFactor": 0.8, "transmissionTexture": image((4, 6, 3))},
            ),
            "leaf": (
                "KHR_materials_diffuse_transmission",
                {
                    "diffuseTransmissionFactor": 0.3,
                    "diffuseTransmissionTexture": image((4, 6, 4)),
                    "diffuseTransmissionColorFactor": [0.2, 0.9, 0.1],
                    "diffuseTransmissionColorTexture": image((5, 3, 3)),
                },
            ),
            "sky": ("KHR_materials_unlit", {}),
        }
        meshes = {}
        for name, (_, fields) in spec.items():
            meshes[name] = g.trimesh.creation.box()
            meshes[name].visual = g.trimesh.visual.TextureVisuals(
                uv=random.random((len(meshes[name].vertices), 2)),
                material=g.trimesh.visual.material.PBRMaterial(
                    roughnessFactor=0.1, unlit=name == "sky" or None, **fields
                ),
            )
        export = g.trimesh.Scene(meshes).export(file_type="glb")
        g.validate_glb(export)
        loaded = g.from_glb_bytes(export).geometry
        again = g.trimesh.Scene(loaded).export(file_type="glb")

        def extensions(data):
            """{mesh name : extensions of its material} in a GLB."""
            header = g.glb_chunks(data)[0]
            return {
                mesh["name"]: header["materials"][mesh["primitives"][0]["material"]].get(
                    "extensions", {}
                )
                for mesh in header["meshes"]
            }

        written, rewritten = extensions(export), extensions(again)
        assert all(
            geometry.visual.material.unlit is None
            for geometry in [loaded["glass"], loaded["leaf"], *rs.geometry.values()]
        ), "lit materials stay lit"
        assert all(
            geometry.visual.material.transmissionFactor is None
            and geometry.visual.material.diffuseTransmissionFactor is None
            for geometry in rs.geometry.values()
        ), "opaque materials have none"
        for name, (extension, fields) in spec.items():
            material = loaded[name].visual.material
            # exactly the spec's fields, and the same object after a reload
            assert list(written[name]) == [extension], f"{name} only {extension}"
            assert set(written[name][extension]) == set(fields), f"{name} every field"
            assert rewritten[name] == written[name], f"{name} rewritten"
            assert material.unlit is (name == "sky" or None), f"{name} unlit"
            for key, value in fields.items():
                # JSON stores floats exactly, and images are lossless PNG
                assert g.np.array_equal(
                    g.np.asarray(getattr(material, key)), g.np.asarray(value)
                ), f"{name} {key}"

    def test_units(self):
        # Trimesh will store units as a GLTF extra if they
        # are defined so check that.

        original = g.get_mesh("pins.glb")

        # export it as a a GLB file
        export = original.export(file_type="glb")
        g.validate_glb(export)

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

        # cameras riding animated nodes round trip on them, and without their
        # optional aspect ratio, which is the viewport's, keep their vertical
        # field of view
        riding = g.dolly_scene("dolly", duration=4.0)

        def no_aspect(tree):
            """Drop the optional aspect ratio and far plane of every camera."""
            for camera in tree["cameras"]:
                camera["perspective"].pop("aspectRatio")
                camera["perspective"].pop("zfar")

        export = riding.export(file_type="glb", tree_postprocessor=no_aspect)
        g.validate_glb(export)
        loaded = g.from_glb_bytes(export)
        # before `loaded.camera`, which makes one if there isn't one
        assert loaded.has_camera, "camera loaded"
        parent = loaded.graph.transforms.parents.get(loaded.camera.name)
        assert parent not in (None, loaded.graph.base_frame), "camera keeps its parent"
        assert g.np.isclose(loaded.camera.fov[1], 45.0), "vertical fov"
        # without `zfar` the projection is infinite, and stays that way
        infinite = g.from_glb_bytes(export).camera
        rewritten = g.glb_chunks(g.from_glb_bytes(export).export(file_type="glb"))[0]
        finite = g.glb_chunks(riding.export(file_type="glb"))[0]["cameras"][0]
        assert infinite.z_far == g.np.inf, "missing zfar is infinite"
        assert g.np.isfinite(infinite.projection).all(), "infinite projection is finite"
        assert all(
            "zfar" not in camera["perspective"] for camera in rewritten["cameras"]
        ), "infinite zfar not written"
        assert finite["perspective"]["zfar"] == riding.camera.z_far, "finite zfar written"
        assert all(
            "aspectRatio" not in camera["perspective"] for camera in rewritten["cameras"]
        ), "missing aspectRatio not invented"
        assert g.np.isclose(
            finite["perspective"]["aspectRatio"],
            g.np.divide(*g.np.tan(g.np.radians(riding.camera.fov) / 2.0)),
        ), "given aspectRatio written"

    def test_gltf_pole(self):
        scene = g.get_mesh("simple_pole.glb")

        # should have multiple primitives
        assert len(scene.geometry) == 11

        export = scene.export(file_type="glb")
        g.validate_glb(export)
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
        g.validate_glb(export)
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
        g.validate_glb(export)
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
        g.validate_glb(export)
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
        g.validate_glb(export)

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

        # joints without weights can't skin so they stay a custom attribute
        box = g.trimesh.creation.box()
        joints = g.np.ones((len(box.vertices), 4), dtype=g.np.uint16)
        box.vertex_attributes["joints"] = joints
        reloaded = g.from_glb_bytes(box.export(file_type="glb"))
        kept = next(iter(reloaded.geometry.values())).vertex_attributes["_joints"]
        assert g.np.array_equal(kept, joints)

        # joints and weights which aren't (n, 4) can't be written as VEC4,
        # which raises when strict and otherwise drops the skinning
        box.vertex_attributes["weights"] = g.np.ones((len(box.vertices), 3))
        with g.pytest.raises(ValueError):
            box.export(file_type="glb")

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
        g.validate_glb(export)
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
        g.validate_glb(export)
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
        g.validate_glb(export)
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
        g.validate_glb(export)
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
        g.validate_glb(export)
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
        g.validate_glb(export)
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
                g.validate_glb(export, name=fn)

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
        g.validate_glb(export)
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
            g.validate_glb(export)

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

        # every gray level through PBR's linear factor and back: a PBR
        # factor is 8 bit linear which merges dark sRGB levels, so the
        # strongest exact predicate is that the stored factor survives
        material = g.trimesh.visual.material
        levels = g.np.arange(256, dtype=g.np.uint8)
        pbr = [
            material.SimpleMaterial(diffuse=[level, level, level, 255]).to_pbr()
            for level in levels
        ]
        stored = g.np.array([current.baseColorFactor for current in pbr])
        diffuse = g.np.array([current.to_simple().diffuse for current in pbr])
        again = [current.to_simple().to_pbr().baseColorFactor for current in pbr]
        # each level which is the only one storing its linear factor
        factors, counts = g.np.unique(stored[:, 0], return_counts=True)
        alone = g.np.isin(stored[:, 0], factors[counts == 1])
        assert g.np.array_equal(again, stored), "linear factor survives"
        assert g.np.array_equal(
            diffuse[alone, :3], g.np.column_stack([levels[alone]] * 3)
        ), "levels storing their own factor are exact"
        assert (diffuse[:, 3] == 255).all(), "alpha exact"
        # the factor is the textbook sRGB decoding of the diffuse, stored
        # in 8 bits so within half a level of it, alpha is never decoded
        rgba = g.np.column_stack([levels, levels[::-1], levels // 2, levels])
        srgb = rgba[:, :3] / 255.0
        exact = g.np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        converted = [material.SimpleMaterial(diffuse=row).to_pbr() for row in rgba]
        factor = g.np.array([current.baseColorFactor for current in converted])
        assert (g.np.abs(factor[:, :3] / 255.0 - exact) <= 0.5 / 255.0 + 1e-9).all(), (
            "factor is linear"
        )
        assert g.np.array_equal(factor[:, 3], rgba[:, 3]), "alpha untouched"
        # an OBJ style material survives GLB with the linear value, not the sRGB one
        box = g.trimesh.creation.box()
        box.visual = g.trimesh.visual.TextureVisuals(
            material=material.SimpleMaterial(diffuse=[188, 92, 24, 255])
        )
        expected = box.visual.material.to_pbr().baseColorFactor
        loaded = next(
            iter(g.roundtrip(box.export(file_type="glb"), "glb").geometry.values())
        )
        assert g.np.array_equal(
            loaded.visual.material.baseColorFactor, expected
        ) and not g.np.array_equal(expected[:3], [188, 92, 24]), "linear factor in glb"

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


def animated_gltf(nodes, channels):
    """
    A glTF header of `nodes` in two scenes, the second holding only the
    last node, with one animation of `channels` as `(node index, path,
    times, values, interpolation)`.
    """
    arrays = [
        g.np.asarray(array, dtype=g.np.float32).reshape((len(array), -1))
        for channel in channels
        for array in channel[2:4]
    ]
    offsets = g.np.cumsum([0] + [array.nbytes for array in arrays])
    data = g.base64.b64encode(b"".join(array.tobytes() for array in arrays))
    kinds = {1: "SCALAR", 3: "VEC3", 4: "VEC4"}
    header = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes) - 1))}, {"nodes": [len(nodes) - 1]}],
        "nodes": nodes,
        "buffers": [
            {
                "byteLength": int(offsets[-1]),
                "uri": "data:application/octet-stream;base64," + data.decode(),
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": int(offset), "byteLength": array.nbytes}
            for offset, array in zip(offsets, arrays)
        ],
        "accessors": [
            {
                "bufferView": index,
                "componentType": 5126,
                "count": len(array),
                "type": kinds[array.shape[1]],
            }
            for index, array in enumerate(arrays)
        ],
        "animations": [
            {
                "channels": [
                    {"sampler": index, "target": {"node": node, "path": path}}
                    for index, (node, path, *_) in enumerate(channels)
                ],
                "samplers": [
                    {"input": 2 * index, "output": 2 * index + 1, "interpolation": kind}
                    for index, (*_, kind) in enumerate(channels)
                ],
            }
        ],
    }
    return header


def test_animation_gltf():
    """
    Animations export as glTF channels which the spec samples to the
    same pose at any time, any glTF channels import as matrices at every
    keyframe of them, and our own exports are a fixed point of loading.
    """
    from trimesh.scene.animation import Animation

    tf = g.trimesh.transformations
    rng = g.np.random.default_rng(3)
    # a nested animated node whose parent carries a `matrix`, which is the
    # case the exporter has to rewrite into parts, and a static node
    scene = g.trimesh.Scene()
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="mover",
        parent_node_name="rig",
        transform=tf.translation_matrix([0, 0, 3]),
    )
    rig = tf.rotation_matrix(0.6, [1, 1, 0], [2, 0, 0])
    scene.graph.update(frame_to="rig", matrix=rig)
    scene.add_geometry(
        g.trimesh.creation.box(),
        node_name="still",
        transform=tf.translation_matrix([5, 0, 0]),
    )
    times = g.np.linspace(0.0, 2.0, 9)
    # the mover turns, slides, and stretches, while the rig only
    # stretches in steps so only its scale is written
    mover = tf.rotation_matrix(times * g.np.pi, [0, 0, 1])
    mover[:, :3, :3] *= rng.uniform(0.5, 2.0, (9, 1, 3))
    mover[:, :3, 3] = rng.normal(size=(9, 3))
    stretch = g.np.tile(g.np.eye(4), (5, 1, 1))
    stretch[:, [0, 1, 2], [0, 1, 2]] = rng.uniform(0.5, 2.0, (5, 3))
    # animations sharing a name are one glTF animation
    scene.animations.extend(
        [
            Animation(
                name="spin",
                nodes=["mover"],
                times=times,
                matrices=mover,
                interpolation="cubic",
            ),
            Animation(
                name="spin",
                nodes=["rig"],
                times=times[::2],
                matrices=rig @ stretch,
                interpolation="step",
            ),
        ]
    )
    built = scene.export(file_type="glb")
    plain = g.trimesh.Scene(g.trimesh.creation.box()).export(file_type="glb")

    # what moves at every time is checked by `assert_glb_roundtrip` below
    header = g.glb_chunks(built)[0]
    animation = header["animations"][0]
    targets = [
        (header["nodes"][c["target"]["node"]]["name"], c["target"]["path"])
        for c in animation["channels"]
    ]
    assert [a["name"] for a in header["animations"]] == ["spin"], "one animation"
    assert sorted(targets) == [
        ("mover", path) for path in ["rotation", "scale", "translation"]
    ] + [("rig", "scale")], "targets"
    assert "animations" not in g.glb_chunks(plain)[0], "plain scene has no animations"
    used = {None}
    for mesh in header["meshes"]:
        for primitive in mesh["primitives"]:
            used.update((*primitive["attributes"].values(), primitive.get("indices")))
    for sampler in animation["samplers"]:
        used.update((sampler["input"], sampler["output"]))
    # every accessor is referenced, and equal ones are written once
    assert used == set(range(len(header["accessors"]))) | {None}, "no stranded accessors"

    # glTF channels on their own keyframes, where node `b` rests stretched
    # and turned and only its rotation moves, and node `c` is in another
    # scene so it isn't loaded or animated
    turns = g.trimesh.util.unitize(rng.normal(size=(6, 4)))
    mixed = [
        (0, "translation", [0.0, 0.5, 2.0], rng.normal(size=(3, 3)), "LINEAR"),
        (0, "rotation", [0.25, 1.0, 1.5, 3.0], turns[:4], "STEP"),
        (0, "scale", [0.0, 1.0, 2.5], rng.uniform(0.5, 2.0, (9, 3)), "CUBICSPLINE"),
        (1, "rotation", [0.5, 0.75, 2.0], turns[3:], "LINEAR"),
        (2, "translation", [0.0, 4.0], rng.normal(size=(2, 3)), "LINEAR"),
    ]
    stepped = [(*channel[:4], "STEP") for channel in mixed if channel[4] != "CUBICSPLINE"]
    linear = [(*channel[:4], "LINEAR") for channel in stepped]
    rest = [
        {"name": "a", "children": [1]},
        {"name": "b", "translation": [1.0, 2.0, 3.0], "scale": [0.5, 2.0, 1.5]},
        {"name": "c", "rotation": turns[5].tolist()},
    ]
    rest[1]["rotation"] = turns[2].tolist()
    sources = {
        "built": scene,
        "box": g.dolly_scene("box"),
        "dolly": g.dolly_scene("dolly"),
        "truck": g.get_mesh("CesiumMilkTruck.glb"),
    }
    for name, channels in {"mixed": mixed, "stepped": stepped, "linear": linear}.items():
        header = animated_gltf(g.copy.deepcopy(rest), channels)
        loaded = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(g.json.dumps(header).encode()),
            file_type="gltf",
        )
        # with something to draw as an empty scene can't be exported
        loaded.add_geometry(g.trimesh.creation.box(), parent_node_name="b")
        sources[name] = loaded
        # each node is in the animation of its smoothest channel, whose
        # keyframes are every keyframe of the channels of its nodes
        kinds = {
            "mixed": {"a": "cubic", "b": "linear"},
            "stepped": {"a": "step", "b": "step"},
            "linear": {"a": "linear", "b": "linear"},
        }[name]
        riding = {
            n: animation for animation in loaded.animations for n in animation.nodes
        }
        assert {
            n: animation.interpolation for n, animation in riding.items()
        } == kinds and all(
            g.np.array_equal(
                animation.times,
                g.np.unique(
                    g.np.float32(
                        [
                            t
                            for c in channels[:-1]
                            if "ab"[c[0]] in animation.nodes
                            for t in c[2]
                        ]
                    )
                ),
            )
            for animation in loaded.animations
        ), f"{name} imported"
        # every loaded channel is the file's value on each of its own
        # keyframes, which a spline also passes through, and holds the
        # first and last outside of them
        for node, path, keys, values, _ in channels[:-1]:
            animation = riding["ab"[node]]
            row = animation.nodes.index("ab"[node])
            values = g.np.float32(values).reshape((len(keys), -1, len(values[0])))
            points = values[:, len(values[0]) // 2]
            times = [*keys, keys[0] - 1.0, keys[-1] + 1.0]
            for time, value in zip(times, [*points, points[0], points[-1]]):
                translation, quaternion, scale = tf.trs_from_matrix(
                    animation.interpolate(time)[row]
                )
                actual = {"translation": translation, "scale": scale}.get(path)
                if path == "rotation":
                    actual = tf.quaternion_matrix(quaternion)[:3, :3]
                    value = tf.quaternion_matrix(value[[3, 0, 1, 2]])[:3, :3]
                assert g.np.allclose(actual, value, atol=1e-6), (
                    f"{name} {node} {path} {time:.3f}"
                )

    for name, source in sources.items():
        g.assert_glb_roundtrip(source, name)


def test_lights_gltf():
    """
    `KHR_lights_punctual` stores lights once in a document array with
    each node referencing one by index, so both halves have to agree.
    """
    lighting = g.trimesh.scene.lighting
    scene = g.trimesh.Scene(g.trimesh.creation.box())
    scene.lights = [
        lighting.DirectionalLight(name="key", color=[255, 244, 224, 255]),
        lighting.PointLight(name="fill", intensity=7.5, radius=12.0),
        lighting.SpotLight(name="rim", innerConeAngle=0.2, outerConeAngle=0.6),
    ]
    for light, x in zip(scene.lights, [1.0, -2.0, 3.0]):
        scene.graph[light.name] = g.trimesh.transformations.translation_matrix([x, 0, 2])
    export = scene.export(file_type="glb")
    g.validate_glb(export, "lights")
    reloaded = g.roundtrip(export, "glb")
    header = g.glb_chunks(export)[0]
    # values and poses round trip in `test_render_cases`, which draws them,
    # and the validator checks every light is declared and referenced
    assert [light.name for light in reloaded.lights] == [
        light.name for light in scene.lights
    ], "light names"
    assert all(
        "KHR_lights_punctual"
        not in (reloaded.graph.transforms.node_data.get(node) or {})
        .get("metadata", {})
        .get("gltf_extensions", {})
        for node in reloaded.graph.nodes
    ) and "KHR_lights_punctual" not in reloaded.metadata.get("gltf_extensions", {}), (
        "not metadata"
    )

    # a scene which never had lights set doesn't grow the extension, and
    # the lights `Scene.lights` makes when asked for are then the scene's
    plain = g.trimesh.Scene(g.trimesh.creation.box())
    assert b"KHR_lights" not in plain.export(file_type="glb"), "plain has no lights"
    defaults = plain.lights
    exported = plain.export(file_type="glb")
    names = [light.name for light in g.roundtrip(exported, "glb").lights]
    assert plain.lights is defaults and names == [light.name for light in defaults], (
        "asked lights are the scene's"
    )
    exported = g.glb_chunks(exported)[0]
    assert not any(
        "extras" in item for item in [*exported["nodes"], *exported["scenes"]]
    ), "asked lights export no extras"

    # a light which doesn't store a color is white per the spec
    header["extensions"]["KHR_lights_punctual"]["lights"][0].pop("color")
    parsed = g.trimesh.exchange.gltf.lights.parse_lights(
        header,
        {index: node.get("name") for index, node in enumerate(header["nodes"])},
        scene_index=0,
        images=None,
        graph=[],
    )
    # the light was warm white, so only the default makes it pure white
    assert parsed[0].name == "key" and (parsed[0].color == 255).all(), (
        "missing color is white"
    )

    NAME = g.trimesh.exchange.gltf.lights.NAME

    def elsewhere(tree):
        """Put a light on the node of a second scene, which isn't loaded."""
        tree["nodes"].append({"name": "elsewhere", "extensions": {NAME: {"light": 0}}})
        tree["scenes"].append({"nodes": [len(tree["nodes"]) - 1]})
        tree["extensions"] = {NAME: {"lights": [{"type": "point"}]}}
        tree["extensionsUsed"] = [NAME]

    box = g.trimesh.Scene(g.trimesh.creation.box())
    other = g.roundtrip(box.export(file_type="glb", tree_postprocessor=elsewhere), "glb")
    assert not other.has_lights, "lights of other scenes not loaded"


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
