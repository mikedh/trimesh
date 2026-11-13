try:
    from . import generic as g
except BaseException:
    import generic as g


def test_unwrap():
    # the same triangles in the same order, with UVs packed into the unit
    # square as charts that never overlap and never fold: charts may be
    # mirrored whole but every triangle in one has the same winding. xatlas
    # (either binding) may pack the very smallest triangles to zero UV area
    if not (g.trimesh.util.has_module("trimkit") or g.trimesh.util.has_module("xatlas")):
        return
    for name in ["bunny.ply", "featuretype.STL"]:
        mesh = g.get_mesh(name, force="mesh")
        unwrapped = mesh.unwrap()
        uv = unwrapped.visual.uv
        assert uv.shape == (len(unwrapped.vertices), 2)
        assert g.np.array_equal(unwrapped.vertices[unwrapped.faces], mesh.triangles)
        assert (uv >= 0.0).all() and (uv <= 1.0).all()

        corners = uv[unwrapped.faces]
        first_edge = corners[:, 1] - corners[:, 0]
        second_edge = corners[:, 2] - corners[:, 0]
        signed = (
            first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
        ) / 2.0
        assert g.np.abs(signed).sum() <= 1.0
        # faces sharing a UV vertex are one chart
        charts = g.trimesh.graph.connected_component_labels(
            g.trimesh.graph.face_adjacency(unwrapped.faces),
            node_count=len(unwrapped.faces),
        )
        for chart in g.np.unique(charts):
            sign = g.np.sign(signed[charts == chart])
            assert len(set(sign[sign != 0])) <= 1

    try:
        from PIL import Image
    except BaseException:
        return
    checkerboard = g.np.kron([[1, 0] * 4, [0, 1] * 4] * 4, g.np.ones((10, 10)))
    image = Image.fromarray((checkerboard * 255).astype(g.np.uint8))
    unwrapped = mesh.unwrap(image=image)
    # make sure image was attached correctly
    assert unwrapped.visual.material.image.size == image.size


class UnwrapTest(g.unittest.TestCase):
    def test_blender_unwrap(self):
        if not g.trimesh.interfaces.blender.exists:
            return
        m = g.get_mesh("rabbit.obj")
        # TODO : verify this returns geometry!
        _todo = g.trimesh.interfaces.blender.unwrap(m)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
