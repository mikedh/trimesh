"""
test_draco.py
-------------

Check `KHR_draco_mesh_compression` in GLTF, which needs the `DracoPy`
package from the `test_more` extra.
"""

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


def glb_header(blob):
    # parse the JSON chunk out of an exported GLB
    length = int.from_bytes(blob[12:16], "little")
    return g.json.loads(blob[20 : 20 + length].decode())


class DracoTest(g.unittest.TestCase):
    def test_export(self):
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
            # draco has no idea what this is so it must be left alone
            custom = r.random((len(colored.vertices), 3)).astype(g.np.float32)
        colored.vertex_attributes["_CustomFloat32Vec3"] = custom

        # both entries are copies so they are byte-identical: the loaded mesh
        # carries vertex normals from the OBJ that a copy recomputes slightly
        # differently, which would make it a genuinely different primitive
        scene = g.trimesh.Scene(
            {"fuze": fuze.copy(), "colored": colored, "duplicate": fuze.copy()}
        )
        # pin `include_normals` so it doesn't depend on what happens to be cached
        kwargs = {"file_type": "glb", "include_normals": True}
        plain = scene.export(**kwargs)
        compressed = scene.export(extension_draco=True, **kwargs)

        # this must actually shrink and must never happen unasked
        assert len(compressed) < len(plain)
        assert b"KHR_draco" not in plain

        validate_glb(compressed, name="draco")
        reloaded = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(compressed), file_type="glb"
        )

        error = {}
        for name, check in reloaded.geometry.items():
            source = scene.geometry[name]
            # exact face equality only holds because we encode `preserve_order`
            assert g.np.array_equal(check.faces, source.faces)
            # vertices must correspond one-to-one, not just match as a set
            error[name] = g.np.abs(check.vertices - source.vertices).max()
            # draco quantizes onto a grid over the bounding box, so half a step
            # is the most any vertex is allowed to move at the exported bit depth
            assert error[name] <= source.extents.max() * 2**-15
            assert check.is_volume == source.is_volume
            assert g.np.isclose(check.volume, source.volume, rtol=1e-3)

        # a box lands exactly on the grid so only check the other side here:
        # more precision than we asked for means the bit depth never arrived
        assert error["fuze"] > scene.geometry["fuze"].extents.max() * 2**-16

        tree = glb_header(compressed)
        assert "KHR_draco_mesh_compression" in tree["extensionsRequired"]
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

        # every accessor draco took over must have given up its buffer
        for primitive in primitives:
            absorbed = [
                primitive["attributes"][name]
                for name in primitive["extensions"]["KHR_draco_mesh_compression"][
                    "attributes"
                ]
            ]
            absorbed.append(primitive["indices"])
            assert not any("bufferView" in tree["accessors"][i] for i in absorbed)

        # the texture survived everything else giving up its buffer
        assert reloaded.geometry["fuze"].visual.material.baseColorTexture is not None

        # an attribute draco never absorbed must keep its own buffer view, as
        # an accessor with neither that nor draco data is defined as all zeros
        attributes = {
            name: primitive["attributes"][name]
            for primitive in primitives
            for name in primitive["attributes"]
            if name.startswith("_")
        }
        assert set(attributes) == {"_CustomFloat32Vec3"}
        assert all("bufferView" in tree["accessors"][i] for i in attributes.values())
        # and it must still be exactly what went in
        assert g.np.allclose(
            reloaded.geometry["colored"].vertex_attributes["_CustomFloat32Vec3"], custom
        )

    def test_shared_accessor(self):
        # accessors are deduplicated by a hash of their data, so a PointCloud
        # built from a compressed mesh's vertices lands on the same accessor.
        # draco must not take that accessor's data away from the point cloud,
        # which is not compressed and has no other way to find its vertices
        if DracoPy is None:
            g.log.info("not testing draco as no `DracoPy`")
            return

        m = g.trimesh.creation.icosphere()
        points = g.trimesh.PointCloud(g.np.array(m.vertices, dtype=g.np.float64))
        scene = g.trimesh.Scene({"mesh": m, "points": points})
        export = scene.export(file_type="glb", extension_draco=True)

        tree = glb_header(export)
        primitives = {
            mesh["name"]: mesh["primitives"][0]
            for mesh in tree["meshes"]
            if "name" in mesh
        }
        # the point cloud is not compressed so it must keep its own buffer:
        # an accessor with no `bufferView` is defined as all zeros
        assert "extensions" not in primitives["points"]
        point_position = primitives["points"]["attributes"]["POSITION"]
        assert "bufferView" in tree["accessors"][point_position]
        # while the mesh it shared vertices with did give its buffer up
        assert (
            "bufferView"
            not in tree["accessors"][primitives["mesh"]["attributes"]["POSITION"]]
        )
        # so they cannot be the same accessor any more
        assert point_position != primitives["mesh"]["attributes"]["POSITION"]

        validate_glb(export, name="draco_shared")
        reloaded = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(export), file_type="glb"
        )
        # the points are uncompressed so they only lost float32 precision,
        # which is far tighter than draco quantization would have been
        assert g.np.allclose(reloaded.geometry["points"].vertices, m.vertices, atol=1e-6)

    def test_unavailable(self):
        # asking for draco and not getting it must never produce a file
        # declaring an extension as required that nothing in it uses, which
        # every conforming loader is obligated to refuse to open
        if DracoPy is None:
            # exporting with `extension_draco` raises without it
            g.log.info("not testing draco as no `DracoPy`")
            return

        from trimesh.exchange.gltf import extensions

        handlers = extensions._handlers["primitive_export"]
        original = handlers["KHR_draco_mesh_compression"]

        def explode(context):
            raise ImportError("no DracoPy")

        # UV, normals, and a texture so every array the handler would have
        # claimed has to find its way back into a buffer view
        source = g.trimesh.Scene({"m": g.get_mesh("fuze.obj")})
        kwargs = {"file_type": "glb", "include_normals": True}

        handlers["KHR_draco_mesh_compression"] = explode
        try:
            export = source.export(extension_draco=True, **kwargs)
        finally:
            handlers["KHR_draco_mesh_compression"] = original

        tree = glb_header(export)
        assert "KHR_draco_mesh_compression" not in tree.get("extensionsRequired", [])
        assert "KHR_draco_mesh_compression" not in tree.get("extensionsUsed", [])
        # every accessor must be backed by real bytes: one with no `bufferView`
        # and no draco data to fill it in is defined as all zeros
        assert all("bufferView" in a for a in tree["accessors"])

        validate_glb(export, name="draco_unavailable")
        check = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(export), file_type="glb"
        ).geometry["m"]
        # the buffers land in a different order than they would have, so compare
        # against a plain export: nothing about the geometry may have changed
        plain = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(source.export(**kwargs)), file_type="glb"
        ).geometry["m"]
        assert g.np.array_equal(check.faces, plain.faces)
        assert g.np.array_equal(check.vertices, plain.vertices)
        assert g.np.array_equal(check.vertex_normals, plain.vertex_normals)
        assert g.np.array_equal(check.visual.uv, plain.visual.uv)
        assert check.visual.material.baseColorTexture is not None


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
