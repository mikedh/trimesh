try:
    from . import generic as g
except BaseException:
    import generic as g


def assert_smooth(mesh, prefix):
    """Same triangles in order with every per-vertex attribute and unit normals."""
    smooth, visual = mesh.smooth_shaded, mesh.visual
    assert len(smooth.faces) == len(mesh.faces), f"{prefix} face count"
    assert g.np.allclose(smooth.vertices[smooth.faces], mesh.vertices[mesh.faces]), (
        f"{prefix} triangles in order"
    )
    assert g.np.allclose(g.np.linalg.norm(smooth.vertex_normals, axis=1), 1), (
        f"{prefix} unit normals"
    )
    # every per-vertex attribute: uv, glTF `COLOR_0`, custom data
    corners = {}
    if visual.kind == "texture":
        after = smooth.visual.vertex_attributes
        corners = {k: (after.get(k), v) for k, v in visual.vertex_attributes.items()}
    elif visual.kind == "vertex":
        corners = {"vertex colors": (smooth.visual.vertex_colors, visual.vertex_colors)}
    elif visual.kind == "face":
        assert (smooth.visual.face_colors == visual.face_colors).all(), (
            f"{prefix} face colors"
        )
    for key, (after, before) in corners.items():
        assert after is not None and g.np.allclose(
            after[smooth.faces], before[mesh.faces]
        ), f"{prefix} {key} in order"


def test_smooth():
    visuals = g.trimesh.visual
    # a cylinder should disconnect its caps from its side exactly
    cylinder = g.trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
    smooth = cylinder.smooth_shaded
    # the normal at every triangle corner
    corner = smooth.vertex_normals[smooth.faces]
    cap = g.np.abs(cylinder.face_normals[:, 2]) > 0.5
    radial = smooth.vertices[smooth.faces][~cap] * [1, 1, 0]
    radial /= g.np.linalg.norm(radial, axis=2)[:, :, None]

    # a fully seamed sphere: every face has its own vertices and
    # its own UV like the worst case texture atlas but every seam is
    # smooth so every coincident vertex should get the same normal
    sphere = g.trimesh.creation.icosphere(subdivisions=3)
    sphere.unmerge_vertices()
    sphere.visual = visuals.TextureVisuals(uv=g.random((len(sphere.vertices), 2)))
    seam = sphere.smooth_shaded
    outward = (seam.vertex_normals * g.trimesh.util.unitize(seam.vertices)).sum(axis=1)

    # a box with random face colors has only sharp edges
    box = g.trimesh.creation.box()
    box.visual.face_colors = g.np.uint8(g.random((12, 4)) * 255)
    flat = box.smooth_shaded

    # a sphere with no seams is one smooth surface, so nothing splits and
    # every corner has the mesh's own angle weighted vertex normal
    round_sphere = g.trimesh.creation.icosphere(subdivisions=3)
    rounded = round_sphere.smooth_shaded

    assert len(smooth.vertices) == len(cylinder.vertices) + 2 * 32, (
        "cylinder splits only at its two rims"
    )
    assert len(flat.vertices) == 24, "box has a vertex for every face corner"
    assert len(rounded.vertices) == len(round_sphere.vertices), "sphere keeps vertices"
    assert g.np.allclose(
        rounded.vertex_normals[rounded.faces],
        round_sphere.vertex_normals[round_sphere.faces],
    ), "sphere keeps its vertex normals"
    assert g.np.allclose(corner[cap], cylinder.face_normals[cap][:, None, :]), (
        "cap normals are exactly +-Z"
    )
    assert g.np.allclose(corner[~cap], radial), "side normals are radial"
    assert (outward > 0.999).all(), "sphere normals are radial"
    assert g.np.allclose(flat.vertex_normals[flat.faces], box.face_normals[:, None, :]), (
        "box normals are face normals"
    )
    assert len(g.trimesh.Trimesh().smooth_shaded.faces) == 0, "empty mesh"
    # `facet_minarea` no longer does anything so passing it warns
    with g.pytest.warns(DeprecationWarning, match="facet_minarea"):
        minarea = g.trimesh.graph.smooth_shade(cylinder, facet_minarea=10.0)
    assert g.np.array_equal(minarea.vertices, smooth.vertices) and g.np.array_equal(
        minarea.faces, smooth.faces
    ), "facet_minarea changes nothing"

    # vertex colors, and a PBR material with per-vertex colors, i.e. glTF `COLOR_0`
    colored = g.trimesh.creation.icosphere()
    colored.visual.vertex_colors = g.random((len(colored.vertices), 4)) * 255
    painted = g.trimesh.creation.box()
    painted.visual = visuals.TextureVisuals(material=visuals.material.PBRMaterial())
    painted.visual.vertex_attributes["color"] = g.np.uint8(g.random((8, 4)) * 255)
    meshes = {"cylinder": cylinder, "sphere": sphere, "box": box, "colored": colored}
    meshes["painted"] = painted
    for name in ["fuze.obj", "chair_model.obj", "featuretype.STL"]:
        meshes[name] = g.get_mesh(name, force="mesh")
    for name, mesh in meshes.items():
        assert_smooth(mesh, name)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_smooth()
