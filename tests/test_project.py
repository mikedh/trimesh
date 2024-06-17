try:
    import generic as g
except ImportError:
    from . import generic as g


def test_projection_already_2D():
    # get a flat part with the flat dimension pointed in +Y
    m = g.get_mesh("1002_tray_bottom.STL")

    # align the mesh with +Z
    m.apply_transform(g.trimesh.geometry.align_vectors([0, 1, 0], [0, 0, 1]))

    # make sure we rotated it correctly so the flat axis
    # is along +Z
    assert m.extents.argmin() == 2

    # get the outline in 3D
    path_3D = m.outline(m.facets[m.facets_area.argmax()])

    # get the projection onto the plane
    path_2D, _to_3D = path_3D.to_planar()

    # just slice off the third column
    path_3D.vertices = path_3D.vertices[:, :2]
    assert path_3D.vertices.shape[1] == 2
    check, to_3D = path_3D.to_planar()
    assert g.np.isclose(check.area, path_2D.area)


if __name__ == "__main__":
    test_projection_already_2D()
