try:
    from . import generic as g
except BaseException:
    import generic as g


def test_process_validate_stale_adjacency():
    """
    https://github.com/mikedh/trimesh/issues/2528

    `process(validate=True)` drops faces under a cache lock, then
    `fix_normals` reads the stale `face_adjacency` and indexes OOB.
    """
    vertices = g.np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [5, 0, 0],
            [6, 0, 0],
            [5, 1, 0],
            [5, 0, 1],
        ],
        dtype=g.np.float64,
    )
    faces = g.np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [4, 5, 6],
            [4, 5, 7],
            [4, 6, 7],
            [5, 6, 7],
        ],
        dtype=g.np.int64,
    )

    mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # populate the face_adjacency cache so the subsequent cache lock
    # hands out the pre-drop version to fix_winding.
    assert mesh.face_adjacency.max() >= len(mesh.faces) - 1

    mesh.process(validate=True, merge_norm=True)

    # after process the adjacency must not reference dropped faces
    assert mesh.face_adjacency.max() < len(mesh.faces)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_process_validate_stale_adjacency()
