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
        hulls = g.get_mesh(name).convex_decomposition()
        assert 1 < len(hulls) <= 64
        for hull in hulls:
            assert len(hull.vertices) <= 64
            if hull.extents.min() > 0.0:
                assert hull.is_volume
                assert hull.is_convex


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_convex_decomposition()
