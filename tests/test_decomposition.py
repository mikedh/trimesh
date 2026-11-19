try:
    from . import generic as g
except BaseException:
    import generic as g


def test_convex_decomposition():
    # every hull is a convex volume within the default 64 vertices. VHACD
    # itself (either binding) can also emit a hull that is exactly flat
    if not (g.trimesh.util.has_module("trimkit") or g.trimesh.util.has_module("vhacdx")):
        return
    for name in ["quadknot.obj", "featuretype.STL"]:
        mesh = g.get_mesh(name)
        # the defaults, then the documented VHACD names on either backend
        for hulls, hull_count, vertex_count in (
            (mesh.convex_decomposition(), 64, 64),
            (mesh.convex_decomposition(maxConvexHulls=4, maxNumVerticesPerCH=16), 4, 16),
        ):
            assert 1 < len(hulls) <= hull_count
            for hull in hulls:
                assert len(hull.vertices) <= vertex_count
                if hull.extents.min() > 0.0:
                    assert hull.is_volume
                    assert hull.is_convex


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_convex_decomposition()
