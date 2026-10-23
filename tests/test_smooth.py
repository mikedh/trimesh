try:
    from . import generic as g
except BaseException:
    import generic as g


def smooth_checks(mesh, prefix):
    """Same triangles in order with every per-vertex attribute and unit normals."""
    smooth, visual = mesh.smooth_shaded, mesh.visual
    checks = {
        "face count": len(smooth.faces) == len(mesh.faces),
        "triangles in order": g.np.allclose(
            smooth.vertices[smooth.faces], mesh.vertices[mesh.faces]
        ),
        "unit normals": g.np.allclose(g.np.linalg.norm(smooth.vertex_normals, axis=1), 1),
    }
    # every per-vertex attribute: uv, glTF `COLOR_0`, custom data
    corners = {}
    if visual.kind == "texture":
        after = smooth.visual.vertex_attributes
        corners = {k: (after.get(k), v) for k, v in visual.vertex_attributes.items()}
    elif visual.kind == "vertex":
        corners = {"vertex colors": (smooth.visual.vertex_colors, visual.vertex_colors)}
    elif visual.kind == "face":
        checks["face colors"] = (smooth.visual.face_colors == visual.face_colors).all()
    for key, (after, before) in corners.items():
        checks[f"{key} in order"] = after is not None and g.np.allclose(
            after[smooth.faces], before[mesh.faces]
        )
    return {f"{prefix} {name}": ok for name, ok in checks.items()}


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

    checks = {
        "cap normals are exactly +-Z": g.np.allclose(
            corner[cap], cylinder.face_normals[cap][:, None, :]
        ),
        "side normals are radial": g.np.allclose(corner[~cap], radial),
        "sphere normals are radial": (outward > 0.999).all(),
        "box normals are face normals": g.np.allclose(
            flat.vertex_normals[flat.faces], box.face_normals[:, None, :]
        ),
        "empty mesh": len(g.trimesh.Trimesh().smooth_shaded.faces) == 0,
    }

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
        checks.update(smooth_checks(mesh, name))
    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_smooth()
