try:
    from . import generic as g
except BaseException:
    import generic as g


def test_quadric_simplification():
    # the faces asked for, and a surface that stays within an output
    # edge of the input both ways: the symmetric Hausdorff distance
    if not (
        g.trimesh.util.has_module("trimkit")
        or g.trimesh.util.has_module("fast_simplification")
    ):
        return

    mesh = g.get_mesh("rabbit.obj")
    assert len(mesh.faces) > 600

    for kwargs, count in [
        ({"percent": 0.5}, len(mesh.faces) // 2),
        ({"face_count": 200}, 200),
        ({"face_count": 60}, 60),
    ]:
        simplified = mesh.simplify_quadric_decimation(**kwargs)
        assert len(simplified.faces) == count
        hausdorff = max(
            g.np.abs(
                g.trimesh.proximity.signed_distance(mesh, simplified.vertices)
            ).max(),
            g.np.abs(
                g.trimesh.proximity.signed_distance(simplified, mesh.vertices)
            ).max(),
        )
        assert hausdorff <= simplified.edges_unique_length.mean()

    # see if aggression does anything
    simplified = mesh.simplify_quadric_decimation(percent=0.25, aggression=5)
    assert len(simplified.faces) > 0


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_quadric_simplification()
