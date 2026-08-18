try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.exchange.gltf import transform as T
from trimesh.exchange.gltf.extensions import GltfLights, registered
from trimesh.scene.animation import RigidAnimation, keyframes_from_matrix
from trimesh.transformations import rotation_matrix

# Khronos' official file validator
# can be installed with the helper script:
# `trimesh/docker/builds/gltf_validator.bash`
_gltf_validator = g.shutil.which("gltf_validator")


def validate_glb(data, name=None):
    """
    Run the Khronos validator on GLB files using
    subprocess.

    Parameters
    ------------
    data : bytes
      GLB export
    name : str or None
      Hint to log.

    Raises
    ------------
    ValueError
      If Khronos validator reports errors.
    """
    if _gltf_validator is None:
        g.log.warning("no gltf_validator!")
        return

    with g.tempfile.NamedTemporaryFile(suffix=".glb") as f:
        f.write(data)
        f.flush()

        # gltf_validator has occasional bugs being run outside
        # of the current working directory
        temp_dir, file_name = g.os.path.split(f.name)
        # run khronos gltf_validator
        report = g.subprocess.run(
            [_gltf_validator, file_name, "-o"], cwd=temp_dir, capture_output=True
        )
        # -o prints JSON to stdout
        content = report.stdout.decode("utf-8")
        returncode = report.returncode

    if returncode != 0:
        g.log.error(f"failed on: `{name}`")
        g.log.error(f"validator: `{content}`")
        g.log.error(f"stderr: `{report.stderr}`")

        raise ValueError("gltf_validator failed")


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

        # a camera is a node in the graph like anything else, so it can be
        # animated by the same machinery: this is what lets a renderer move
        # the camera without being told how
        lighting = g.trimesh.scene.lighting
        scene.lights = [
            lighting.DirectionalLight(name="key", color=[255, 244, 224, 255]),
            lighting.PointLight(name="fill", intensity=7.5, radius=12.0),
            lighting.SpotLight(name="rim", innerConeAngle=0.2, outerConeAngle=0.6),
        ]
        for light, x in zip(scene.lights, [1.0, -2.0, 3.0]):
            scene.graph[light.name] = g.trimesh.transformations.translation_matrix(
                [x, 0, 2]
            )

        times = g.np.linspace(0.0, 2.0, 24)
        poses = rotation_matrix(times * g.np.pi, [0, 0, 1]) @ scene.camera_transform
        scene.animations.append(
            RigidAnimation(
                frame_to=scene.camera.name, times=times, matrices=poses, name="orbit"
            )
        )

        reloaded = g.roundtrip(scene.export(file_type="glb"), "glb")

        # the animation has to come back pointing at the camera the file
        # actually has, not a dangling node named like the one it had
        orbit = next(a for a in reloaded.animations if a.name == "orbit")
        assert orbit.frame_to == reloaded.camera.name
        assert reloaded.camera.name in reloaded.graph.nodes
        # float32 is what GLTF stores keyframes as
        assert g.np.allclose(orbit.at(times), poses, atol=1e-5)

        # lights survive with everything that distinguishes one from another
        assert [type(x).__name__ for x in reloaded.lights] == [
            type(x).__name__ for x in scene.lights
        ]
        for before, after in zip(scene.lights, reloaded.lights):
            assert before.name == after.name
            assert g.np.allclose(before.color, after.color, atol=1)
            assert g.np.isclose(before.intensity, after.intensity)
            assert before.radius == after.radius
            # a light without its transform is not a light
            assert g.np.allclose(
                scene.graph[before.name][0], reloaded.graph[after.name][0]
            )
        spot = reloaded.lights[-1]
        assert g.np.isclose(spot.innerConeAngle, 0.2)
        assert g.np.isclose(spot.outerConeAngle, 0.6)

        # a scene which never had lights set shouldn't grow the extension,
        # as `scene.lights` would happily invent a pair on being asked
        plain = g.trimesh.Scene(g.trimesh.creation.box())
        assert not plain.has_lights
        assert "KHR_lights_punctual" not in plain.export(file_type="gltf")[
            "model.gltf"
        ].decode("utf8")

        # the extension is handled by the registry rather than inline, which
        # is what lets anything else hook the document and node level too
        assert GltfLights.NAME in registered("scene")
        assert GltfLights.NAME in registered("scene_export")

        # write it onto a tree by hand, with no exporter in between: a node
        # refers to a light by its *position* in the document array, which is
        # the one contract joining the two halves
        tree = {"nodes": [{"name": L.name} for L in scene.lights]}
        # something already on a node, which the merge must not stomp
        tree["nodes"][0]["extensions"] = {"MY_extension": {"keep": True}}
        node_index = {L.name: i for i, L in enumerate(scene.lights)}
        GltfLights(scene.lights).to_gltf(tree, node_index)

        stored = tree["extensions"][GltfLights.NAME]["lights"]
        assert len(stored) == len(scene.lights)
        for i, light in enumerate(scene.lights):
            node = tree["nodes"][node_index[light.name]]
            assert node["extensions"][GltfLights.NAME]["light"] == i
            assert stored[i]["name"] == light.name
        assert tree["nodes"][0]["extensions"]["MY_extension"] == {"keep": True}

        # and reading that tree back is an exact inverse
        names = {i: L.name for i, L in enumerate(scene.lights)}
        for before, after in zip(scene.lights, GltfLights.from_gltf(tree, names).lights):
            assert type(before) is type(after)
            assert before.name == after.name
            assert g.np.allclose(before.color, after.color, atol=1)
            assert g.np.isclose(before.intensity, after.intensity)
            assert before.radius == after.radius
        assert g.np.isclose(
            GltfLights.from_gltf(tree, names).lights[-1].outerConeAngle, 0.6
        )

        # two nodes may share one array entry, which has to give two lights
        shared = {
            "extensions": {GltfLights.NAME: {"lights": [stored[1]]}},
            "nodes": [
                {"extensions": {GltfLights.NAME: {"light": 0}}},
                {"extensions": {GltfLights.NAME: {"light": 0}}},
            ],
        }
        both = GltfLights.from_gltf(shared, {0: "a", 1: "b"}).lights
        assert [L.name for L in both] == ["a", "b"]

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

        scene_extensions = {"mesh_ext": {"ext_data": 1.23}}
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
        mesh_extensions = {"mesh_ext": {"ext_data": 1.23}}
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
        node_extensions = {"mesh_ext": {"ext_data": 1.23}}
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

        header = g.json.loads(export[20 : 20 + int.from_bytes(export[12:16], "little")])
        # a primitive with no material inherits GLTF's default, which is
        # fully metallic and fully rough: every mesh needs a real one
        primitives = [p for mesh in header["meshes"] for p in mesh["primitives"]]
        assert all("material" in p for p in primitives)
        # colors ride as COLOR_0 which GLTF multiplies against the base
        # color, so one white material serves every primitive
        assert all("COLOR_0" in p["attributes"] for p in primitives)
        assert len(header["materials"]) == 1
        pbr = header["materials"][0]["pbrMetallicRoughness"]
        assert g.np.allclose(pbr["baseColorFactor"], 1.0)
        assert pbr["metallicFactor"] == 0.0

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

        # `to_pbr` has to write `metallicFactor` even when it is zero, as
        # leaving it out means GLTF's default of 1.0 and every simple
        # material would export as a mirror
        pbr = simple.to_pbr()
        assert pbr.metallicFactor == 0.0
        # and roughness is still derived from the specular exponent
        assert g.np.isclose(pbr.roughnessFactor, (2 / (simple.glossiness + 2)) ** 0.25)

        # an explicit value wins over both defaults
        asked = simple.to_pbr(metallic=0.9, roughness=0.15)
        assert asked.metallicFactor == 0.9
        assert asked.roughnessFactor == 0.15

        # and lands in the file rather than being dropped on the way
        box = g.trimesh.creation.box()
        box.visual = g.trimesh.visual.TextureVisuals(material=asked)
        export = box.export(file_type="glb")
        header = g.json.loads(export[20 : 20 + int.from_bytes(export[12:16], "little")])
        stored = header["materials"][0]["pbrMetallicRoughness"]
        assert stored["metallicFactor"] == 0.9
        assert stored["roughnessFactor"] == 0.15

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


def test_animation_gltf():
    """
    Exported animations should roundtrip and satisfy the parts of the
    GLTF spec which the schema alone can't check.
    """
    scene, spin = animated_scene()
    export = scene.export(file_type="glb")
    validate_glb(export, "animation")
    reloaded = g.roundtrip(export, "glb")

    assert len(reloaded.animations) == len(scene.animations)

    for original in scene.animations:
        # find the animation which came back for this node and name
        match = [
            b
            for b in reloaded.animations
            if b.frame_to == original.frame_to and b.name == original.name
        ]
        assert len(match) == 1
        other = match[0]
        assert other.interpolation == original.interpolation

        # the whole edge has to survive, not just the node it targets.
        # a loader which dropped `frame_from` would still sample
        # identically here and pass every other assertion below
        assert other.frame_from == original.frame_from
        # and it has to be the edge the reloaded graph actually has
        assert reloaded.graph.transforms.parents[other.frame_to] == other.frame_from

        # sampling rather than comparing raw arrays means this holds
        # even if channels were split or merged onto a new time base
        # and still catches quaternion order, transposition, and
        # a misaligned time base all at once.
        # sample the middle of each keyframe interval: GLTF requires
        # animation input accessors be float32, so a query landing
        # within an epsilon of a boundary can legitimately step to a
        # different keyframe than the float64 original would
        query = (original.times[1:] + original.times[:-1]) / 2.0
        assert g.np.allclose(original.at(query), other.at(query), atol=1e-5)

        if original.interpolation == "linear":
            # a continuous animation can be checked much more densely
            dense = g.np.linspace(
                original.times[0], original.times[-1], len(original) * 7
            )
            assert g.np.allclose(original.at(dense), other.at(dense), atol=1e-5)

    # a rigid input has to come back rigid: a pure rotation is orthonormal
    # with a determinant of exactly +1, which fails if any scale or shear
    # leaked in from the decomposition
    spun = next(a for a in reloaded.animations if a.name == "spin")
    rigid = spun.at(g.np.linspace(spun.times[0], spun.times[-1], 101))
    rotation = rigid[:, :3, :3]
    assert g.np.allclose(rotation @ rotation.transpose(0, 2, 1), g.np.eye(3), atol=1e-5)
    assert g.np.allclose(g.np.linalg.det(rotation), 1.0, atol=1e-5)
    # the bottom row of a homogeneous transform is never touched
    assert g.np.allclose(rigid[:, 3, :], [0, 0, 0, 1])

    # ------------------------------------------------------- the spec
    tree, _ = g.trimesh.exchange.gltf._create_gltf_structure(scene)
    # every animation we defined should be grouped by name
    assert len(tree["animations"]) == len({a.name for a in scene.animations})

    header = g.json.loads(
        g.trimesh.exchange.gltf.export_gltf(scene, embed_buffers=True)["model.gltf"]
    )
    blobs = accessor_values(header)

    targeted = set()
    for entry in header["animations"]:
        # the spec requires every {node, path} target in one animation
        # be unique, which a strict loader will refuse a file over
        pairs = [(c["target"]["node"], c["target"]["path"]) for c in entry["channels"]]
        assert len(pairs) == len(set(pairs))

        for channel in entry["channels"]:
            targeted.add(channel["target"]["node"])
            sampler = entry["samplers"][channel["sampler"]]

            times = blobs[sampler["input"]].reshape(-1)
            # the spec requires min/max on an animation input accessor
            accessor = header["accessors"][sampler["input"]]
            assert g.np.allclose(accessor["min"], times.min())
            assert g.np.allclose(accessor["max"], times.max())
            # keyframe times must be increasing
            assert (g.np.diff(times) >= 0).all()

            values = blobs[sampler["output"]]
            assert len(values) == len(times)

            if channel["target"]["path"] == "rotation":
                # exported rotations must be unit quaternions
                assert g.np.allclose(g.np.linalg.norm(values, axis=1), 1.0, atol=1e-6)
                # adjacent keyframes must share a hemisphere or a viewer
                # interpolating between them takes the long way around
                # and the animation will visibly jerk
                assert (g.np.sum(values[1:] * values[:-1], axis=1) >= -1e-6).all()

    # the spec forbids a matrix on any node an animation targets
    assert len(targeted) > 0
    for index in targeted:
        node = header["nodes"][index]
        assert "matrix" not in node
        # the matrix must have been replaced by an equivalent TRS rather
        # than simply dropped: every link is built 3 units along Z
        assert g.np.allclose(node["translation"], [0, 0, 3])

    # node names should survive so animations can be matched back up
    assert spin.frame_to in {n.get("name") for n in header["nodes"]}

    # nothing should be left in the file which nothing points at: a
    # channel dropped for being static used to strand its time accessor
    used = {
        index
        for entry in header["animations"]
        for sampler in entry["samplers"]
        for index in (sampler["input"], sampler["output"])
    }
    for mesh in header["meshes"]:
        for primitive in mesh["primitives"]:
            used.update(primitive["attributes"].values())
            if "indices" in primitive:
                used.add(primitive["indices"])
    assert used == set(range(len(header["accessors"])))

    # an animation whose channels all match the node's static pose has
    # nothing to say, and shouldn't leave an accessor behind saying it
    static = g.trimesh.Scene(g.trimesh.creation.box())
    static.animations.append(
        RigidAnimation(
            frame_to=static.graph.nodes_geometry[0],
            times=g.np.linspace(0, 1, 5),
            matrices=g.np.tile(g.np.eye(4), (5, 1, 1)),
        )
    )
    quiet = g.json.loads(
        g.trimesh.exchange.gltf.export_gltf(static, embed_buffers=True)["model.gltf"]
    )
    assert "animations" not in quiet
    assert len(quiet["accessors"]) == len(
        {
            index
            for mesh in quiet["meshes"]
            for primitive in mesh["primitives"]
            for index in list(primitive["attributes"].values())
            + [primitive.get("indices")]
            if index is not None
        }
    )

    # ------------------------------------------------------ cubic export
    # the tangents are what makes this worth checking: an exporter which
    # writes the values but drops them, or a loader which narrows the mode
    # to LINEAR, still round trips the keyframes perfectly and only
    # differs in between them
    random = g.np.random.default_rng(11)
    times = g.np.linspace(0.0, 2.0, 9)
    matrices = rotation_matrix(times * g.np.pi * 0.5, [0, 0, 1])
    matrices[:, :3, 3] = random.uniform(-2.0, 2.0, (len(times), 3))

    keyframes = keyframes_from_matrix(times, matrices)
    keyframes["translation_in"] = random.uniform(-2.0, 2.0, (len(times), 3))
    keyframes["translation_out"] = random.uniform(-2.0, 2.0, (len(times), 3))
    keyframes["quaternion_in"] = random.uniform(-0.3, 0.3, (len(times), 4))
    keyframes["quaternion_out"] = random.uniform(-0.3, 0.3, (len(times), 4))

    curved = g.trimesh.Scene()
    curved.add_geometry(g.trimesh.creation.box(), node_name="spinner")
    curved.animations.append(
        RigidAnimation(
            frame_to="spinner", keyframes=keyframes, name="cubic", interpolation="cubic"
        )
    )

    export = curved.export(file_type="glb")
    # the validator checks a CUBICSPLINE output accessor is exactly
    # three times its input, which a malformed export would fail
    validate_glb(export, "animation_cubic")

    reloaded = g.roundtrip(export, "glb")
    assert len(reloaded.animations) == 1
    other = reloaded.animations[0]
    assert other.interpolation == "cubic"
    assert len(other) == len(times)

    # every keyframe field has to survive, tangents included. GLTF stores
    # these as float32 so the tolerance is the storage, not the math
    for field in keyframes.dtype.names:
        assert g.np.allclose(other.keyframes[field], keyframes[field], atol=1e-6), field

    # and the sampled path has to agree densely, not just at keyframes
    dense = g.np.linspace(times[0], times[-1], len(times) * 11)
    original = curved.animations[0]
    assert g.np.allclose(original.at(dense), other.at(dense), atol=1e-5)

    # a linear reload would agree at the keyframes and nowhere else,
    # so confirm the two modes are actually distinguishable here
    linear = RigidAnimation(
        frame_to="spinner", keyframes=keyframes.copy(), interpolation="linear"
    )
    assert g.np.abs(original.at(dense) - linear.at(dense)).max() > 1e-2

    # ------------------------------------------------- a real assembly
    assembly = g.get_mesh("cycloidal.3DXML")
    assert len(assembly.animations) == 0
    # a scene which was never animated has to export exactly as it did
    # before: no empty animations array, and nodes still using a matrix
    plain, _ = g.trimesh.exchange.gltf._create_gltf_structure(assembly)
    assert "animations" not in plain
    assert any("matrix" in n for n in plain["nodes"])
    assert not any(
        k in n for n in plain["nodes"] for k in ("translation", "rotation", "scale")
    )

    # spin every camshaft instance about the drive axis
    nodes = [n for n in assembly.graph.nodes if str(n).startswith("camshaft")]
    assert len(nodes) > 0

    times = g.np.linspace(0.0, 2.0, 17)
    # (n, 4, 4) stack of rotations about Z, built without a python loop
    spinning = rotation_matrix(times * g.np.pi, [0, 0, 1])

    for node in nodes:
        # walking a loaded graph, so the edge has to be asked for
        parent = assembly.graph.transforms.parents[node]
        local = assembly.graph.get(frame_to=node, frame_from=parent)[0]
        assembly.animations.append(
            RigidAnimation(
                frame_to=node,
                frame_from=parent,
                times=times,
                matrices=spinning @ local,
                name="spin",
            )
        )

    export = assembly.export(file_type="glb")
    validate_glb(export, "cycloidal-animated")
    reloaded = g.roundtrip(export, "glb")

    assert len(reloaded.animations) == len(nodes)
    # every animation should have come from the single named group
    assert {a.name for a in reloaded.animations} == {"spin"}

    lookup = {a.frame_to: a for a in reloaded.animations}
    for original in assembly.animations:
        other = lookup[original.frame_to]
        assert g.np.allclose(
            original.at(times), other.at(times), atol=1e-4 * assembly.scale
        )


def test_animation_load():
    """
    Loading animations from hand-built GLTF: splines, unsupported
    channels, and channels which don't share a time base.
    """
    # --------------------------------------------------- a plain spline
    count = 4
    times = g.np.linspace(0.0, 3.0, count).astype("<f4")
    half = times.astype(g.np.float64) * g.np.pi / 6.0

    # CUBICSPLINE output holds (in-tangent, value, out-tangent) per keyframe
    # so it is three times as long as the input, and only the middle is used
    values = g.np.zeros((count, 3, 4), dtype="<f4")
    values[:, 1] = g.np.column_stack(
        [g.np.zeros(count), g.np.zeros(count), g.np.sin(half), g.np.cos(half)]
    )

    header = gltf_animated(
        [times, values],
        samplers=[{"input": 0, "output": 1, "interpolation": "CUBICSPLINE"}],
        channels=[
            {"sampler": 0, "target": {"node": 0, "path": "rotation"}},
            # morph target weights aren't supported and must be skipped
            # rather than raising or corrupting the node
            {"sampler": 0, "target": {"node": 0, "path": "weights"}},
        ],
        node="spinner",
        name="cubic",
    )
    scene = g.roundtrip(g.json.dumps(header).encode(), "gltf")

    assert len(scene.animations) == 1
    current = scene.animations[0]
    assert current.name == "cubic"
    assert len(current) == count
    # the spline must be kept as a spline rather than narrowed
    assert current.interpolation == "cubic"

    # a cubic reproduces its keyframes exactly at the keyframe times
    expected = g.trimesh.transformations.quaternion_matrix(
        g.np.column_stack([g.np.cos(half), g.np.zeros((count, 2)), g.np.sin(half)])
    )
    assert g.np.allclose(current.at(times.astype(g.np.float64)), expected, atol=1e-6)

    # these tangents are all zero, which makes the curve ease in and out of
    # every keyframe. that is a different path than a constant-rate slerp,
    # so a silent downgrade to LINEAR anywhere would collapse this to zero
    linear = RigidAnimation(
        frame_to=current.frame_to,
        keyframes=current.keyframes.copy(),
        interpolation="linear",
    )
    query = g.np.linspace(0.0, 3.0, 97)
    assert g.np.abs(current.at(query) - linear.at(query)).max() > 1e-3

    # and the rotation is still a rotation the whole way along
    rotation = current.at(query)[:, :3, :3]
    assert g.np.allclose(rotation @ rotation.transpose(0, 2, 1), g.np.eye(3), atol=1e-8)

    # ------------------------------------------------ mixed time bases
    # channels of one node may reference different input accessors, and
    # have to land on a shared time base for a single keyframe array,
    # which is the only path in the loader that resamples anything
    t_move = g.np.array([0.0, 1.0, 2.0], dtype="<f4")
    t_spin = g.np.array([0.0, 0.5, 1.5, 2.0], dtype="<f4")

    channels = [
        {"sampler": 0, "target": {"node": 0, "path": "translation"}},
        {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
    ]

    # translation ramps at 1 unit/second, rotation at pi/4 radians/second
    move = g.np.column_stack(
        [t_move.astype(g.np.float64), g.np.zeros((len(t_move), 2))]
    ).astype("<f4")
    half = t_spin.astype(g.np.float64) * g.np.pi / 8.0
    # GLTF orders quaternions `xyzw`
    spin = g.np.column_stack(
        [g.np.zeros((len(t_spin), 2)), g.np.sin(half), g.np.cos(half)]
    ).astype("<f4")

    scene = g.roundtrip(
        g.json.dumps(
            gltf_animated(
                [t_move, t_spin, move, spin],
                samplers=[{"input": 0, "output": 2}, {"input": 1, "output": 3}],
                channels=channels,
                name="mixed",
            )
        ).encode(),
        "gltf",
    )
    assert len(scene.animations) == 1
    current = scene.animations[0]
    # the keyframes are the union of both channels' times
    assert g.np.allclose(current.times, [0.0, 0.5, 1.0, 1.5, 2.0])

    # both channels are constant-rate, so resampling onto a superset of
    # their own knots has to be exact and both stay analytic everywhere.
    # a resample which dropped interpolation would still match at the
    # keyframes and fail in between, which is why this is sampled densely
    query = g.np.linspace(0.0, 2.0, 197)
    sampled = current.at(query)

    assert g.np.allclose(sampled[:, 0, 3], query, atol=1e-6)
    assert g.np.allclose(sampled[:, 1:3, 3], 0.0, atol=1e-6)

    # the rotation is `query * pi / 4` about Z the whole way
    expected = g.trimesh.transformations.quaternion_matrix(
        g.np.column_stack(
            [
                g.np.cos(query * g.np.pi / 8.0),
                g.np.zeros((len(query), 2)),
                g.np.sin(query * g.np.pi / 8.0),
            ]
        )
    )
    assert g.np.allclose(sampled[:, :3, :3], expected[:, :3, :3], atol=1e-6)

    # ------------------------------- a spline forced onto foreign times
    # it cannot stay a spline, but it has to be resampled *along* its real
    # curve. dropping the tangents before resampling loses the whole curve
    # while still reproducing every keyframe, so this only fails in between

    # collinear values with strong opposing tangents, so the spline bulges
    # well away from the straight line those keyframes would otherwise draw
    move = g.np.zeros((len(t_move), 3, 3), dtype="<f4")
    move[:, 1, 0] = t_move  # value
    move[:, 0, 0] = -6.0  # in-tangent
    move[:, 2, 0] = 6.0  # out-tangent
    spin = g.np.tile(g.np.array([0.0, 0.0, 0.0, 1.0], dtype="<f4"), (len(t_spin), 1))

    scene = g.roundtrip(
        g.json.dumps(
            gltf_animated(
                [t_move, t_spin, move, spin],
                samplers=[
                    {"input": 0, "output": 2, "interpolation": "CUBICSPLINE"},
                    {"input": 1, "output": 3},
                ],
                channels=channels,
                name="mixed",
            )
        ).encode(),
        "gltf",
    )
    current = scene.animations[0]
    # a spline resampled onto foreign times has no tangents left to keep
    assert current.interpolation == "linear"
    assert g.np.allclose(current.times, [0.0, 0.5, 1.0, 1.5, 2.0])

    # the analytic Hermite curve these keyframes and tangents describe
    query = g.np.linspace(0.0, 2.0, 121)
    lower = g.np.clip(g.np.searchsorted([0.0, 1.0, 2.0], query), 1, 2) - 1
    blend = (query - lower).reshape((-1, 1))
    squared, cubed = blend**2, blend**3
    truth = (
        (2 * cubed - 3 * squared + 1) * lower.reshape((-1, 1))
        + (cubed - 2 * squared + blend) * 6.0
        + (-2 * cubed + 3 * squared) * (lower + 1).reshape((-1, 1))
        + (cubed - squared) * -6.0
    ).ravel()

    # the spline has to bulge well off the straight line, or this proves
    # nothing about whether the tangents were carried through
    assert g.np.abs(truth - query).max() > 0.5

    sampled = current.at(query)[:, 0, 3]
    # resampling along the real curve tracks it far better than the
    # straight line that dropping the tangents would collapse it to
    assert g.np.abs(sampled - truth).max() < 0.5
    assert g.np.abs(sampled - truth).max() < g.np.abs(query - truth).max() / 2.0


def test_transform():
    """
    A GLTF node dict and a matrix should be two spellings of one transform.
    """
    tf = g.trimesh.transformations

    # non-uniform scale throughout and a mirror every fifth, which can't
    # be a unit quaternion so it has to land in the scale instead
    random = g.np.random.default_rng(0)
    matrices = tf.random_rotation_matrix(num=100, seed=0)
    scale = random.uniform(0.2, 4.0, (100, 3))
    scale[::5, 0] *= -1.0
    matrices[:, :3, :3] *= scale.reshape((-1, 1, 3))
    matrices[:, :3, 3] = random.uniform(-10.0, 10.0, (100, 3))
    assert (g.np.linalg.det(matrices[::5, :3, :3]) < 0).all()

    # decompose the whole stack the way the exporter does, from the
    # column-major flattening GLTF stores
    trs = T.trs_from_gltf_matrices(matrices.transpose(0, 2, 1).reshape((-1, 16)))
    # which has to agree with composing them straight back
    assert g.np.allclose(tf.tqs_matrix(*trs), matrices)

    # writing each one out as a node and reading it back as an exact
    # inverse pins the TRS split, the XYZW ordering, and the omitted
    # defaults all with one predicate
    for i, matrix in enumerate(matrices):
        node = {}
        T.node_from_trs([part[i] for part in trs], node)
        assert g.np.allclose(tf.tqs_matrix(*T.trs_from_node(node)), matrix)
    # a transposed matrix would survive a symmetric one, so make sure
    # the column-major handling above was actually exercised
    assert not g.np.allclose(matrices[0], matrices[0].T)

    def as_node(matrix):
        # write a single matrix out the way the exporter does
        node = {}
        T.node_from_trs(
            [p[0] for p in T.trs_from_gltf_matrices(matrix.T.reshape(-1))], node
        )
        return node

    # a component at its default is dropped even when the others aren't:
    # a pure rotation has a scale of exactly one and no translation
    assert set(as_node(tf.rotation_matrix(0.7, [1.0, 2.0, 3.0]))) == {"rotation"}
    # anything already at the GLTF default is omitted rather than written
    assert as_node(g.np.eye(4)) == {}
    # and a node with no keys at all reads back as identity
    assert g.np.allclose(tf.tqs_matrix(*T.trs_from_node({})), g.np.eye(4))

    # ---------------------------------------------------------- unwind
    with g.RandomSeed() as random:
        quaternion = tf.random_quaternion(num=50, seed=0)
        # flip a random half of them into the opposite hemisphere, which
        # is the same rotation but the long way around when interpolated
        flip = random.random(len(quaternion)) > 0.5
        flipped = quaternion * g.np.where(flip, -1.0, 1.0).reshape((-1, 1))

    unwound = T.unwind(flipped)

    # every adjacent pair now shares a hemisphere
    assert (g.np.sum(unwound[1:] * unwound[:-1], axis=1) >= 0).all()
    # and every quaternion is still the same rotation it started as
    assert g.np.allclose(g.np.abs(g.trimesh.util.diagonal_dot(unwound, flipped)), 1.0)
    # which is a real change, i.e. the input actually needed unwinding
    assert not (g.np.sum(flipped[1:] * flipped[:-1], axis=1) >= 0).all()


def accessor_values(header):
    """
    Decode every accessor in an embedded GLTF header into numpy arrays.

    Parameters
    ------------
    header : dict
      GLTF header with a single embedded base64 buffer.

    Returns
    ----------
    values : list
      Numpy array for each accessor.
    """
    from trimesh.exchange.gltf import _dtypes, _shapes

    # every buffer is embedded as a base64 data URI
    buffers = [g.base64.b64decode(b["uri"].split(",", 1)[1]) for b in header["buffers"]]

    values = []
    for accessor in header["accessors"]:
        view = header["bufferViews"][accessor["bufferView"]]
        blob = buffers[view["buffer"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        dtype = g.np.dtype(_dtypes[accessor["componentType"]])
        # how many values make up one element
        per_count = int(g.np.prod(_shapes[accessor["type"]]))
        length = accessor["count"] * per_count * dtype.itemsize
        data = g.np.frombuffer(blob[start : start + length], dtype=dtype)
        values.append(data.reshape((accessor["count"], per_count)))

    return values


def gltf_animated(arrays, samplers, channels, node="mover", name="anim"):
    """
    Build a minimal single-node GLTF header holding one animation.

    Every array is packed into its own buffer view and accessor in the
    order passed, with the shape deciding the type and count: a
    CUBICSPLINE output is an `(n, 3, d)` stack of in-tangent, value, and
    out-tangent which flattens to the `3n` elements the spec asks for.

    Parameters
    ------------
    arrays : list of array
      Data for one accessor each, in index order.
    samplers : list of dict
      GLTF samplers, referencing accessors by index.
    channels : list of dict
      GLTF channels, referencing samplers by index.
    node : str
      Name of the single node every channel targets.
    name : str
      Name of the animation.

    Returns
    ----------
    header : dict
      A loadable GLTF file with one embedded buffer.
    """
    # GLTF stores animation data as little-endian float32
    stacked = [g.np.asanyarray(a, dtype="<f4") for a in arrays]
    # a 1D array is a scalar channel, otherwise the last axis is the
    # component count and everything before it flattens into elements
    packed = [a.reshape((-1, 1 if a.ndim == 1 else a.shape[-1])) for a in stacked]

    blob = b"".join(a.tobytes() for a in packed)
    offsets = g.np.concatenate([[0], g.np.cumsum([a.nbytes for a in packed])])

    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": node}],
        "buffers": [
            {
                "byteLength": len(blob),
                "uri": "data:application/octet-stream;base64,"
                + g.base64.b64encode(blob).decode(),
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": int(o), "byteLength": int(a.nbytes)}
            for o, a in zip(offsets, packed)
        ],
        "accessors": [
            {
                "bufferView": i,
                "componentType": 5126,
                "count": len(a),
                "type": {1: "SCALAR", 3: "VEC3", 4: "VEC4"}[a.shape[1]],
                # the spec requires min and max on an animation input
                **(
                    {"min": [float(a.min())], "max": [float(a.max())]}
                    if a.shape[1] == 1
                    else {}
                ),
            }
            for i, a in enumerate(packed)
        ],
        "animations": [{"name": name, "samplers": samplers, "channels": channels}],
    }


def animated_scene():
    """
    Build a small scene with a nested graph and a few animations.

    Returns
    ----------
    scene : trimesh.Scene
      Scene with `animations` populated.
    spin : RigidAnimation
      The animation driving the deepest node.
    """
    tf = g.trimesh.transformations

    scene = g.trimesh.Scene()
    # a nested chain so local transforms compound down the graph, with the
    # base frame at the front so `chain[i]` is the parent of `chain[i + 1]`
    # and every animated edge is just an adjacent pair
    chain = ["world", "link_0", "link_1", "link_2"]
    for parent, node in g.itertools.pairwise(chain):
        scene.add_geometry(
            g.trimesh.creation.box(extents=[1, 1, 3]),
            node_name=node,
            parent_node_name=parent,
            transform=tf.translation_matrix([0, 0, 3]),
        )

    times = g.np.linspace(0.0, 2.0, 25)

    # a pure rotation on the deepest node
    spin = RigidAnimation(
        frame_to=chain[3],
        frame_from=chain[2],
        times=times,
        matrices=rotation_matrix(times * g.np.pi, [0, 0, 1]),
        name="spin",
    )
    scene.animations.append(spin)

    # translation and non-uniform scale on another node, and a
    # differently named animation so more than one group is exported
    wobble = g.np.tile(g.np.eye(4), (len(times), 1, 1))
    wobble[:, :3, 3] = g.np.column_stack(
        [g.np.sin(times * 2.0), g.np.zeros(len(times)), 3.0 + g.np.cos(times)]
    )
    wobble[:, :3, :3] *= g.np.column_stack(
        [1.0 + 0.25 * g.np.sin(times * 3.0), g.np.ones((len(times), 2))]
    ).reshape((-1, 1, 3))
    scene.animations.append(
        RigidAnimation(
            frame_to=chain[1],
            frame_from=chain[0],
            times=times,
            matrices=wobble,
            name="wobble",
        )
    )

    # a stepped animation to exercise the other interpolation mode
    stepped = g.np.tile(g.np.eye(4), (len(times), 1, 1))
    stepped[:, 2, 3] = 3.0 + g.np.sin(times)
    scene.animations.append(
        RigidAnimation(
            frame_to=chain[2],
            frame_from=chain[1],
            times=times,
            matrices=stepped,
            name="step",
            interpolation="step",
        )
    )

    return scene, spin


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
