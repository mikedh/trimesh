"""
test_draco.py
-------------

Check `KHR_draco_mesh_compression` in GLTF, which needs the `DracoPy`
package from the `test_more` extra.
"""

import base64

try:
    from . import generic as g
    from .test_gltf import validate_glb
except BaseException:
    import generic as g
    from test_gltf import validate_glb

try:
    import DracoPy
except BaseException:
    DracoPy = None


def draco_gltf(mesh):
    """
    Wrap a mesh as a draco-compressed GLTF without going through our exporter.

    Building the asset here rather than committing one keeps a third-party
    encoder in the loop for the decode path.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry, must have UV coordinates.

    Returns
    ----------
    file_obj : io.BytesIO
      A GLTF file with all geometry inside one draco buffer.
    """
    # pass every optional attribute so `POSITION` isn't draco's first attribute:
    # a decoder that ignores the id mapping and trusts ordering will put the
    # vertices into the color slot and produce garbage
    with g.RandomSeed() as r:
        colors = (r.random((len(mesh.vertices), 4)) * 255).astype(g.np.uint8)

    # GLTF stores V the other way up
    uv = g.np.array(mesh.visual.uv, dtype=g.np.float64)
    uv[:, 1] = 1.0 - uv[:, 1]

    buffer = DracoPy.encode(
        points=mesh.vertices.astype(g.np.float32),
        faces=mesh.faces.astype(g.np.uint32),
        colors=colors,
        tex_coord=uv,
        normals=g.np.asarray(mesh.vertex_normals, dtype=g.np.float64),
        preserve_order=True,
        quantization_bits=24,
        compression_level=6,
    )

    # ask draco which id it gave each attribute
    ids = {a["attribute_type"]: a["unique_id"] for a in DracoPy.decode(buffer).attributes}
    attributes = {
        "POSITION": ids[DracoPy.AttributeType.POSITION],
        "NORMAL": ids[DracoPy.AttributeType.NORMAL],
        "COLOR_0": ids[DracoPy.AttributeType.COLOR],
        "TEXCOORD_0": ids[DracoPy.AttributeType.TEX_COORD],
    }

    # accessors carry no `bufferView` as all of the data is inside draco
    tree = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "extensionsRequired": ["KHR_draco_mesh_compression"],
        "extensionsUsed": ["KHR_draco_mesh_compression"],
        # UV is only kept as UV if something is textured by it
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 1,
                            "NORMAL": 2,
                            "COLOR_0": 3,
                            "TEXCOORD_0": 4,
                        },
                        "indices": 0,
                        "material": 0,
                        "mode": 4,
                        "extensions": {
                            "KHR_draco_mesh_compression": {
                                "bufferView": 0,
                                "attributes": attributes,
                            }
                        },
                    }
                ]
            }
        ],
        "accessors": [
            {"componentType": 5125, "count": mesh.faces.size, "type": "SCALAR"},
            {"componentType": 5126, "count": len(mesh.vertices), "type": "VEC3"},
            {"componentType": 5126, "count": len(mesh.vertices), "type": "VEC3"},
            {"componentType": 5121, "count": len(mesh.vertices), "type": "VEC4"},
            {"componentType": 5126, "count": len(mesh.vertices), "type": "VEC2"},
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(buffer)}],
        "buffers": [
            {
                "byteLength": len(buffer),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(buffer).decode(),
            }
        ],
    }
    return g.io.BytesIO(g.json.dumps(tree).encode())


def glb_header(blob):
    """
    Pull the JSON header out of an exported GLB.

    Parameters
    ------------
    blob : bytes
      A GLB file.

    Returns
    ----------
    header : dict
      The parsed GLTF tree.
    """
    length = int.from_bytes(blob[12:16], "little")
    return g.json.loads(blob[20 : 20 + length].decode())


def referenced_views(tree):
    """
    Collect every `bufferView` index anything in a tree points at.

    Parameters
    ------------
    tree : dict
      A GLTF header.

    Returns
    ----------
    referenced : set
      Indexes of buffer views which are actually used.
    """
    referenced = set()
    queue = [tree]
    while queue:
        current = queue.pop()
        if isinstance(current, dict):
            if isinstance(current.get("bufferView"), int):
                referenced.add(current["bufferView"])
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return referenced


class DracoTest(g.unittest.TestCase):
    def test_decode(self):
        # a draco-compressed primitive must come back as the original geometry
        if DracoPy is None:
            g.log.info("not testing draco as no `DracoPy`")
            return

        m = g.get_mesh("fuze.obj")
        r = g.trimesh.load_mesh(draco_gltf(m), file_type="gltf")

        # quantization is the only thing allowed to move a vertex
        assert g.np.array_equal(r.faces, m.faces)
        assert g.np.allclose(r.vertices, m.vertices, atol=1e-6)
        # volume is signed and depends on winding so it catches the
        # permutations and axis swaps a bounding box comparison won't
        assert g.np.isclose(r.volume, m.volume, rtol=1e-4)
        # UV would survive being swapped with normals so check it directly
        assert g.np.allclose(r.visual.uv, m.visual.uv, atol=1e-6)

    def test_decode_no_handler_warns(self):
        # the same file with no decoder registered must warn rather than
        # silently produce a mesh full of zeros
        if DracoPy is None:
            g.log.info("not testing draco as no `DracoPy`")
            return

        from trimesh.exchange.gltf import extensions as ext

        blob = draco_gltf(g.get_mesh("fuze.obj")).getvalue()
        handler = ext._handlers["primitive_preprocess"].pop("KHR_draco_mesh_compression")
        try:
            with self.assertLogs("trimesh", level="WARNING") as cm:
                r = g.trimesh.load_mesh(g.io.BytesIO(blob), file_type="gltf")
        finally:
            ext._handlers["primitive_preprocess"]["KHR_draco_mesh_compression"] = handler

        assert g.np.allclose(r.vertices, 0.0)
        assert any("KHR_draco_mesh_compression" in line for line in cm.output)

    def test_export_roundtrip(self):
        # exported draco must be smaller and reload as the same mesh
        if DracoPy is None:
            g.log.info("not testing draco as no `DracoPy`")
            return

        m = g.get_mesh("featuretype.STL")
        plain = m.export(file_type="glb")
        compressed = m.export(file_type="glb", extension_draco=True)

        # an untextured mesh is all geometry so this must actually shrink
        assert len(compressed) < len(plain)

        # only does anything where the khronos validator is installed
        validate_glb(compressed, name="draco")

        r = g.trimesh.load_mesh(
            g.trimesh.util.wrap_as_stream(compressed), file_type="glb"
        )

        # exact face equality only holds because we encode with `preserve_order`
        assert g.np.array_equal(r.faces, m.faces)
        assert len(r.vertices) == len(m.vertices)
        # vertices must correspond one-to-one, not just as a set
        assert g.np.allclose(r.vertices, m.vertices, atol=m.scale * 1e-3)
        assert g.np.isclose(r.volume, m.volume, rtol=1e-3)

        tree = glb_header(compressed)
        assert "KHR_draco_mesh_compression" in tree["extensionsRequired"]

        # every accessor draco took over must have given up its buffer
        for mesh in tree["meshes"]:
            for primitive in mesh["primitives"]:
                extension = primitive["extensions"]["KHR_draco_mesh_compression"]
                absorbed = [
                    primitive["attributes"][name] for name in extension["attributes"]
                ]
                absorbed.append(primitive["indices"])
                assert not any("bufferView" in tree["accessors"][i] for i in absorbed)

        # if the uncompressed data were still in the file it would be orphaned
        assert referenced_views(tree) == set(range(len(tree["bufferViews"])))

    def test_off_by_default(self):
        # draco must never appear unless it was explicitly asked for
        m = g.get_mesh("featuretype.STL")
        assert b"KHR_draco" not in m.export(file_type="glb")
        assert not any(b"KHR_draco" in v for v in m.export(file_type="gltf").values())
        assert b"KHR_draco" not in g.trimesh.Scene({"m": m}).export(file_type="glb")

    def test_quantization(self):
        # the quantization constant must actually reach draco
        if DracoPy is None:
            g.log.info("not testing draco as no `DracoPy`")
            return

        from trimesh.exchange.gltf import extensions as ext

        m = g.get_mesh("featuretype.STL")
        deviation = {}
        size = {}
        original = ext._DRACO_QUANT
        try:
            for bits in [12, 16, 24]:
                ext._DRACO_QUANT = bits
                blob = m.export(file_type="glb", extension_draco=True)
                r = g.trimesh.load_mesh(
                    g.trimesh.util.wrap_as_stream(blob), file_type="glb"
                )
                # order is preserved at every setting so this is a per-vertex error
                deviation[bits] = g.np.abs(r.vertices - m.vertices).max()
                size[bits] = len(blob)
        finally:
            ext._DRACO_QUANT = original

        # each bit halves the quantization step so error must fall geometrically:
        # a constant that never reached the encoder would make these equal
        assert deviation[12] > deviation[16] * 8
        assert deviation[16] > deviation[24] * 8
        # and precision is not free
        assert size[12] < size[16] < size[24]

    def test_export_scene(self):
        # textures, vertex colors, and duplicated geometry in one scene
        if DracoPy is None:
            g.log.info("not testing draco as no `DracoPy`")
            return

        fuze = g.get_mesh("fuze.obj")
        colored = g.trimesh.creation.box()
        with g.RandomSeed() as r:
            colored.visual.vertex_colors = (
                r.random((len(colored.vertices), 4)) * 255
            ).astype(g.np.uint8)

        # both entries are copies so they are byte-identical: the loaded mesh
        # carries vertex normals from the OBJ that a copy recomputes slightly
        # differently, which would make it a genuinely different primitive
        scene = g.trimesh.Scene(
            {"fuze": fuze.copy(), "colored": colored, "duplicate": fuze.copy()}
        )
        # pin `include_normals` so it doesn't depend on what happens to be cached
        compressed = scene.export(
            file_type="glb", extension_draco=True, include_normals=True
        )
        validate_glb(compressed, name="draco scene")
        reloaded = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(compressed), file_type="glb"
        )

        for name, check in reloaded.geometry.items():
            source = scene.geometry[name]
            assert g.np.array_equal(check.faces, source.faces)
            assert g.np.allclose(
                check.vertices, source.vertices, atol=source.scale * 1e-3
            )

        tree = glb_header(compressed)
        primitives = [p for m in tree["meshes"] for p in m["primitives"]]

        # every mesh must be compressed: identical geometry shares accessors so
        # a handler which consumed them would silently skip the second copy
        views = [
            p["extensions"]["KHR_draco_mesh_compression"]["bufferView"]
            for p in primitives
        ]
        assert len(views) == len(scene.geometry)

        # `fuze` and `duplicate` are identical so they must share one buffer
        assert len(set(views)) == len(scene.geometry) - 1

        # the texture survived the buffers being renumbered around it
        assert reloaded.geometry["fuze"].visual.material.baseColorTexture is not None
        assert referenced_views(tree) == set(range(len(tree["bufferViews"])))


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
