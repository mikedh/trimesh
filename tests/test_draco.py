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

needs_draco = g.pytest.mark.skipif(DracoPy is None, reason="no `DracoPy`")

# pin `include_normals` so nothing depends on what happens to be cached
kwargs = {"file_type": "glb", "include_normals": True}


def glb_tree(blob):
    # parse the JSON chunk out of an exported GLB
    return g.json.loads(blob[20 : 20 + int.from_bytes(blob[12:16], "little")].decode())


def draco_scene():
    # a texture, vertex colors, a custom attribute draco can't absorb,
    # duplicated geometry, and a point cloud sharing a mesh's vertices
    fuze = g.get_mesh("fuze.obj")
    colored = g.trimesh.creation.box()
    count = len(colored.vertices)
    with g.RandomSeed() as r:
        colored.visual.vertex_colors = (r.random((count, 4)) * 255).astype(g.np.uint8)
        colored.vertex_attributes["_Custom"] = r.random((count, 3)).astype(g.np.float32)
    # both `fuze` are copies so they are byte-identical: the loaded mesh carries
    # vertex normals from the OBJ that a copy recomputes slightly differently
    return g.trimesh.Scene(
        {
            "fuze": fuze.copy(),
            "duplicate": fuze.copy(),
            "colored": colored,
            "points": g.trimesh.PointCloud(colored.vertices.copy()),
        }
    )


class DracoTest(g.unittest.TestCase):
    @needs_draco
    def test_export(self):
        scene = draco_scene()
        plain = scene.export(**kwargs)
        compressed = scene.export(extension_draco=True, **kwargs)

        # this must actually shrink and must never happen unasked
        assert len(compressed) < len(plain)
        assert b"KHR_draco" not in plain
        validate_glb(compressed, name="draco")

        tree = glb_tree(compressed)
        assert "KHR_draco_mesh_compression" in tree["extensionsRequired"]
        primitives = {m["name"]: m["primitives"][0] for m in tree["meshes"]}

        # accessors are deduplicated by a hash of their data, so uncompressed
        # the point cloud lands on the same one as the box it was built from:
        # without that this has nothing to say about what draco takes over
        shared = {
            m["name"]: m["primitives"][0]["attributes"]["POSITION"]
            for m in glb_tree(plain)["meshes"]
        }
        assert shared["points"] == shared["colored"]

        # every mesh must be compressed and the two identical ones must share a
        # buffer: they also share accessors, so a handler which consumed those
        # would have silently skipped the second copy
        views = [
            p["extensions"]["KHR_draco_mesh_compression"]["bufferView"]
            for name, p in primitives.items()
            if name != "points"
        ]
        assert len(views) == 3
        assert len(set(views)) == 2

        # an accessor with neither a `bufferView` nor draco data is defined as
        # all zeros, so the accessors which gave theirs up must be exactly the
        # ones draco absorbed: not the point cloud sharing the box's vertices,
        # and not a custom attribute draco has no idea what to do with
        absorbed = set()
        for p in primitives.values():
            draco = p.get("extensions", {}).get("KHR_draco_mesh_compression")
            if draco is None:
                continue
            absorbed.update(p["attributes"][name] for name in draco["attributes"])
            absorbed.add(p["indices"])
        assert absorbed == {
            i for i, a in enumerate(tree["accessors"]) if "bufferView" not in a
        }
        # and the attribute draco skipped was really in there to be skipped
        assert "_Custom" in primitives["colored"]["attributes"]

        reloaded = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(compressed), file_type="glb"
        )
        error = {}
        for name, check in reloaded.geometry.items():
            source = scene.geometry[name]
            if name == "points":
                # uncompressed so it only lost float32 precision, which is far
                # tighter than draco quantization would have been
                assert g.np.allclose(check.vertices, source.vertices, atol=1e-6)
                continue
            # exact face equality only holds because we encode `preserve_order`
            assert g.np.array_equal(check.faces, source.faces)
            # vertices must correspond one-to-one, not just match as a set
            error[name] = g.np.abs(check.vertices - source.vertices).max()
            # draco quantizes onto a grid over the bounding box, so half a step
            # is the most any vertex may move at the exported bit depth
            assert error[name] <= source.extents.max() * 2**-15
            assert check.is_volume == source.is_volume
            assert g.np.isclose(check.volume, source.volume, rtol=1e-3)

        # a box lands exactly on the grid so only check the other side here:
        # more precision than we asked for means the bit depth never arrived
        assert error["fuze"] > scene.geometry["fuze"].extents.max() * 2**-16
        # the texture and the attribute draco skipped both survived
        assert reloaded.geometry["fuze"].visual.material.baseColorTexture is not None
        assert g.np.allclose(
            reloaded.geometry["colored"].vertex_attributes["_Custom"],
            scene.geometry["colored"].vertex_attributes["_Custom"],
        )

    @needs_draco
    def test_unavailable(self):
        # asking for draco and not getting it must never produce a file
        # declaring an extension as required that nothing in it uses, which
        # every conforming loader is obligated to refuse to open
        from trimesh.exchange.gltf import extensions

        handlers = extensions._handlers["primitive_export"]
        original = handlers["KHR_draco_mesh_compression"]

        def explode(context):
            raise ImportError("no DracoPy")

        # UV, normals, and a texture so every array the handler would have
        # claimed has to find its way back into a buffer view
        source = g.get_mesh("fuze.obj")
        handlers["KHR_draco_mesh_compression"] = explode
        try:
            export = g.trimesh.Scene({"m": source}).export(extension_draco=True, **kwargs)
        finally:
            handlers["KHR_draco_mesh_compression"] = original

        tree = glb_tree(export)
        assert "KHR_draco_mesh_compression" not in tree.get("extensionsRequired", [])
        assert "KHR_draco_mesh_compression" not in tree.get("extensionsUsed", [])
        # every accessor must be backed by real bytes: one with no `bufferView`
        # and no draco data to fill it in is defined as all zeros
        assert all("bufferView" in a for a in tree["accessors"])

        validate_glb(export, name="draco_unavailable")
        check = g.trimesh.load_scene(
            g.trimesh.util.wrap_as_stream(export), file_type="glb"
        ).geometry["m"]
        # so the geometry only lost the float32 a plain export would have
        assert g.np.array_equal(check.faces, source.faces)
        assert g.np.allclose(check.vertices, source.vertices, atol=1e-6)
        assert g.np.allclose(check.visual.uv, source.visual.uv, atol=1e-6)
        assert check.visual.material.baseColorTexture is not None


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
