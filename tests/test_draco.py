"""
test_draco.py
-------------

Check `KHR_draco_mesh_compression` in GLTF, which needs `DracoPy`
from the `test_more` extra.
"""

from importlib.util import find_spec

try:
    from . import generic as g
    from .test_gltf import validate_glb
except BaseException:
    import generic as g
    from test_gltf import validate_glb

needs_draco = g.pytest.mark.skipif(find_spec("DracoPy") is None, reason="no `DracoPy`")
# pin `include_normals` so nothing depends on what happens to be cached
kwargs = {"file_type": "glb", "include_normals": True}


def export(scene, **more):
    # export a GLB and return it alongside its parsed JSON chunk
    blob = scene.export(**kwargs, **more)
    return blob, g.json.loads(blob[20 : 20 + int.from_bytes(blob[12:16], "little")])


def absorbed(tree):
    # the accessors draco took over, which must be exactly the accessors with no
    # `bufferView` as one with neither that nor draco data is defined as all zeros
    taken = set()
    for primitive in [p for m in tree["meshes"] for p in m["primitives"]]:
        draco = primitive.get("extensions", {}).get("KHR_draco_mesh_compression")
        if draco is not None:
            taken.update(primitive["attributes"][name] for name in draco["attributes"])
            taken.add(primitive["indices"])
    assert taken == {i for i, a in enumerate(tree["accessors"]) if "bufferView" not in a}
    return taken


def draco_scene():
    # a texture, vertex colors, a custom attribute draco can't absorb, duplicated
    # geometry, and a point cloud sharing a mesh's vertices
    fuze = g.get_mesh("fuze.obj")
    box = g.trimesh.creation.box()
    with g.RandomSeed() as r:
        box.visual.vertex_colors = (r.random((8, 4)) * 255).astype(g.np.uint8)
        box.vertex_attributes["_Custom"] = r.random((8, 3)).astype(g.np.float32)
    # both `fuze` are copies so they are byte-identical: the loaded mesh carries
    # vertex normals from the OBJ that a copy recomputes slightly differently
    return g.trimesh.Scene(
        {
            "fuze": fuze.copy(),
            "duplicate": fuze.copy(),
            "box": box,
            "points": g.trimesh.PointCloud(box.vertices.copy()),
        }
    )


@needs_draco
def test_export():
    from trimesh.exchange.gltf import extensions

    scene = draco_scene()
    plain, before = export(scene)
    blob, tree = export(scene, extension_draco=True)
    validate_glb(blob, name="draco")

    # it must shrink, must never happen unasked, and must be declared
    assert len(blob) < len(plain) and b"KHR_draco" not in plain
    assert "KHR_draco_mesh_compression" in tree["extensionsRequired"]
    # the registration is what dispatches it, as the exporter runs every
    # handler for the scope rather than naming this one
    assert "KHR_draco_mesh_compression" in extensions.registered("primitive_export")

    # accessors are deduplicated by content, so `fuze` and `duplicate` share one
    # and so do `box` and the point cloud built from its vertices: without that
    # the compressed export has nothing to say about what draco takes over
    shared = [m["primitives"][0]["attributes"]["POSITION"] for m in before["meshes"]]
    assert len(shared) == 4 and len(set(shared)) == 2
    # every mesh must still be compressed, into two buffers rather than three
    views = [
        p["extensions"]["KHR_draco_mesh_compression"]["bufferView"]
        for m in tree["meshes"]
        for p in m["primitives"]
        if "extensions" in p
    ]
    assert len(views) == 3 and len(set(views)) == 2
    # 4 accessors each for `fuze` and `box`: the point cloud sharing vertices
    # with `box` and the `_Custom` draco can't absorb must not be among them
    assert len(absorbed(tree)) == 8

    reloaded = g.trimesh.load_scene(g.trimesh.util.wrap_as_stream(blob), file_type="glb")
    error = {
        name: g.np.abs(mesh.vertices - scene.geometry[name].vertices).max()
        for name, mesh in reloaded.geometry.items()
    }
    # draco quantizes onto a grid over the bounding box so half a step is the most
    # any vertex may move, and getting more precision than we asked for on `fuze`
    # would mean the bit depth never arrived at all
    assert all(e <= scene.geometry[n].extents.max() * 2**-15 for n, e in error.items())
    assert error["fuze"] > scene.geometry["fuze"].extents.max() * 2**-16
    # exact face equality only holds because we encode `preserve_order`
    assert all(
        g.np.array_equal(mesh.faces, scene.geometry[name].faces)
        for name, mesh in reloaded.geometry.items()
        if name != "points"
    )
    # the texture and the attribute draco skipped both survived
    assert reloaded.geometry["fuze"].visual.material.baseColorTexture is not None
    assert g.np.allclose(
        reloaded.geometry["box"].vertex_attributes["_Custom"],
        scene.geometry["box"].vertex_attributes["_Custom"],
    )


@needs_draco
def test_unavailable():
    # asking for draco and not getting it must never produce a file declaring an
    # extension as required that nothing uses, which loaders must refuse to open
    from trimesh.exchange.gltf import extensions

    def explode(context):
        raise ImportError("no DracoPy")

    handlers = extensions._handlers["primitive_export"]
    original = handlers["KHR_draco_mesh_compression"]
    scene = draco_scene()
    try:
        handlers["KHR_draco_mesh_compression"] = explode
        blob, tree = export(scene, extension_draco=True)
    finally:
        handlers["KHR_draco_mesh_compression"] = original

    validate_glb(blob, name="draco_unavailable")
    assert "KHR_draco" not in str(tree.get("extensionsUsed", []))
    assert "KHR_draco" not in str(tree.get("extensionsRequired", []))
    # nothing absorbed anything so every array must have found its way back
    assert not absorbed(tree)
    # leaving geometry which only lost the float32 a plain export would have
    reloaded = g.trimesh.load_scene(g.trimesh.util.wrap_as_stream(blob), file_type="glb")
    assert all(
        g.np.allclose(mesh.vertices, scene.geometry[name].vertices, atol=1e-6)
        for name, mesh in reloaded.geometry.items()
    )
    assert reloaded.geometry["fuze"].visual.material.baseColorTexture is not None


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_export()
    test_unavailable()
