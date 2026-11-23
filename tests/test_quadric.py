try:
    from . import generic as g
except BaseException:
    import generic as g


def test_quadric_simplification():
    if not g.trimesh.util.has_module("fast_simplification"):
        return

    m = g.get_mesh("rabbit.obj")
    assert len(m.faces) > 600

    # should be about half as large
    a = m.simplify_quadric_decimation(percent=0.5)
    assert g.np.isclose(len(a.faces), len(m.faces) // 2, rtol=0.2)

    # should have the requested number of faces
    a = m.simplify_quadric_decimation(face_count=200)
    assert len(a.faces) == 200

    # see if aggression does anything
    a = m.simplify_quadric_decimation(percent=0.25, aggression=5)
    assert len(a.faces) > 0


def test_scene_quadric_simplification_respects_face_count():
    if not g.trimesh.util.has_module("fast_simplification"):
        return

    small = g.trimesh.creation.box()
    exact = g.trimesh.creation.icosphere(subdivisions=0)
    large = g.trimesh.creation.icosphere(subdivisions=1)
    scene = g.trimesh.Scene({"small": small, "exact": exact, "large": large})

    scene.simplify_quadric_decimation(face_count=len(exact.faces))

    assert scene.geometry["small"] is small
    assert scene.geometry["exact"] is exact
    assert len(scene.geometry["large"].faces) == len(exact.faces)


if __name__ == "__main__":
    test_quadric_simplification()
