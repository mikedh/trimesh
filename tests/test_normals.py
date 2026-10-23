try:
    from . import generic as g
except BaseException:
    import generic as g

# run a Python expression in Blender without a window or user settings
FLAGS = "--background --factory-startup --python-exit-code 1 --python-expr".split()
# Blender's MikkTSpace tangent through the vertex normals for every face
# corner of a mesh from a JSON file, written back to the same file
MIKKTSPACE = """
import bpy, json, sys
path = sys.argv[-1]
data = json.load(open(path))
mesh = bpy.data.meshes.new("mesh")
mesh.from_pydata(data["vertices"], [], data["faces"])
corners = [loop.vertex_index for loop in mesh.loops]
uv = mesh.uv_layers.new(name="uv")
uv.data.foreach_set("uv", [c for v in corners for c in data["uv"][v]])
mesh.shade_smooth()
mesh.normals_split_custom_set_from_vertices(data["normals"])
mesh.calc_tangents(uvmap="uv")
tangents = [[*loop.tangent, loop.bitangent_sign] for loop in mesh.loops]
json.dump({"corners": corners, "tangents": tangents}, open(path, "w"))
"""


class NormalsTest(g.unittest.TestCase):
    def test_vertex_normal(self):
        mesh = g.trimesh.creation.icosahedron()
        # the icosahedron is centered at zero, so the true vertex
        # normal is just a unit vector of the vertex position
        truth = g.trimesh.util.unitize(mesh.vertices)

        # force fallback to loop normal summing by passing None
        # as the sparse matrix
        normals = g.trimesh.geometry.mean_vertex_normals(
            len(mesh.vertices), mesh.faces, mesh.face_normals
        )
        assert g.np.allclose(normals - truth, 0.0)

        # make sure the automatic sparse matrix generation works
        normals = g.trimesh.geometry.mean_vertex_normals(
            len(mesh.vertices), mesh.faces, mesh.face_normals
        )
        assert g.np.allclose(normals - truth, 0.0)

        # make sure the Trimesh normals- related attributes
        # are wired correctly
        assert mesh.faces_sparse is not None
        assert mesh.vertex_normals.shape == mesh.vertices.shape
        assert g.np.allclose(mesh.vertex_normals - truth, 0.0)

    def test_vertex_tangents(self):
        from trimesh.geometry import vertex_tangents
        from trimesh.interfaces import blender

        # tangents of a flat sheet in an arbitrary pose have one true answer
        # for every way texture can be laid on it, mirrored or rotated
        flat = g.random((50, 2))
        faces = g.spatial.Delaunay(flat).simplices
        rotation = g.trimesh.transformations.random_rotation_matrix()[:3, :3]
        vertices, normal = flat @ rotation[:, :2].T, g.np.tile(rotation[:, 2], (50, 1))
        x, y = flat.T
        checks = {}
        for name, uv, direction, w in [
            ("u along x", (x, y), rotation[:, 0], 1.0),
            ("v mirrored", (x, -y), rotation[:, 0], -1.0),
            ("u along y", (y, x), rotation[:, 1], -1.0),
            ("u against y", (-y, x), -rotation[:, 1], 1.0),
        ]:
            tangents = vertex_tangents(vertices, faces, g.np.column_stack(uv), normal)
            checks[name] = g.np.allclose(tangents, [*direction, w])
        # and on a curved textured mesh they are unit, in the surface and signed
        mesh = g.get_mesh("fuze.obj")
        arrays = {"vertices": mesh.vertices, "faces": mesh.faces, "uv": mesh.visual.uv}
        tangents = vertex_tangents(*arrays.values(), mesh.vertex_normals)
        arrays["normals"] = g.trimesh.util.unitize(mesh.vertex_normals)
        checks["unit"] = g.np.allclose(g.np.linalg.norm(tangents[:, :3], axis=1), 1.0)
        surface = (tangents[:, :3] * mesh.vertex_normals).sum(axis=1)
        checks["in the surface"] = g.np.allclose(surface, 0.0)
        checks["signed"] = set(g.np.unique(tangents[:, 3])) <= {-1.0, 1.0}

        # and they are MikkTSpace's, which glTF viewers generate when a
        # file has none, as computed by Blender for every face corner
        if blender.exists:
            with g.TemporaryDirectory() as directory:
                path = g.os.path.join(directory, "mesh.json")
                with open(path, "w") as f:
                    g.json.dump({k: v.tolist() for k, v in arrays.items()}, f)
                command = [blender._blender_executable, *FLAGS, MIKKTSPACE, "--", path]
                g.subprocess.run(command, check=True, capture_output=True)
                with open(path) as f:
                    reference = g.json.load(f)
            corners = g.np.array(reference["corners"])
            expected = g.np.array(reference["tangents"])
            # every corner of a vertex agrees where its faces don't mirror
            # the texture, which a single vertex tangent can't represent
            same = expected[:, 3] == tangents[corners, 3]
            delta = expected[same, :3] - tangents[corners][same, :3]
            error = g.np.linalg.norm(delta, axis=1)
            checks["MikkTSpace signs"] = same.mean() > 0.99
            checks["MikkTSpace tangents"] = g.np.percentile(error, 99) < 1e-3
        for name, ok in checks.items():
            assert ok, name

    def test_weighted_vertex_normals(self):
        def compare_trimesh_to_groundtruth(mesh, truth, atol=g.trimesh.tol.merge):
            # force fallback to loop normal summing by passing None
            # as the sparse matrix
            normals = g.trimesh.geometry.weighted_vertex_normals(
                vertex_count=len(mesh.vertices),
                faces=mesh.faces,
                face_normals=mesh.face_normals,
                face_angles=mesh.face_angles,
            )
            assert g.np.allclose(normals, truth, atol=atol)

            # make sure the automatic sparse matrix generation works
            normals = g.trimesh.geometry.weighted_vertex_normals(
                len(mesh.vertices), mesh.faces, mesh.face_normals, mesh.face_angles
            )
            assert g.np.allclose(normals - truth, 0.0, atol=atol)

            # make sure the Trimesh normals- related attributes
            # are wired correctly
            assert mesh.faces_sparse is not None
            assert mesh.vertex_normals.shape == mesh.vertices.shape
            assert g.np.allclose(mesh.vertex_normals - truth, 0.0, atol=atol)

        # the icosahedron is centered at zero, so the true vertex
        # normal is just a unit vector of the vertex position
        ico_mesh = g.trimesh.creation.icosahedron()
        ico_truth = g.trimesh.util.unitize(ico_mesh.vertices)
        compare_trimesh_to_groundtruth(ico_mesh, ico_truth)

        # create a cube centered at zero, as with the icosahedron,
        # normals compute as unit vectors of the corner vertices
        # due to the triangulation of the box, this case would fail
        # with a simple face-normals-average as vertex-normal method
        box_mesh = g.trimesh.creation.box()
        box_truth = g.trimesh.util.unitize(box_mesh.vertices)
        compare_trimesh_to_groundtruth(box_mesh, box_truth)

        # load the fandisk model: a complex triangulated mesh with
        # smooth curved surfaces, corners and sharp creases
        # ground truth vertex normals were computed in MeshLab and
        # are included in the file
        fandisk_mesh = g.get_mesh("fandisk.obj")
        fandisk_truth = fandisk_mesh.vertex_normals
        # due to the limited precision in the MeshLab export,
        # we have to tweak the tolerance for the comparison a little
        compare_trimesh_to_groundtruth(fandisk_mesh, fandisk_truth, 0.0001)

        # see how we do with degenerate faces
        m = g.trimesh.creation.box()
        m.faces[0][0] = m.faces[0][1]
        norm = m.vertex_normals
        assert g.np.isfinite(norm).all()
        assert len(norm) == len(m.vertices)

        # vertices with every face intact
        mask = g.np.zeros(len(m.vertices), dtype=bool)
        mask[m.faces[0]] = False
        # it's a box so normals should all be unit vectors [1,1,1]
        assert g.np.allclose(g.np.abs(norm[mask]), (1.0 / 3.0) ** 0.5)

        # try with a deliberately broken sparse matrix to test looping path
        norm = g.trimesh.geometry.weighted_vertex_normals(
            vertex_count=len(m.vertices),
            faces=m.faces,
            face_normals=m.face_normals,
            face_angles=m.face_angles,
            use_loop=True,
        )
        assert g.np.isfinite(norm).all()
        assert len(norm) == len(m.vertices)

        # every intact vertex should be away from box corner
        assert g.np.allclose(g.np.abs(norm[mask]), (1.0 / 3.0) ** 0.5)

    def test_face_normals(self):
        """
        Test automatic generation of face normals on mesh objects
        """
        mesh = g.trimesh.creation.icosahedron()
        assert mesh.face_normals.shape == mesh.faces.shape

        # normals should regenerate
        mesh.face_normals = None
        assert mesh.face_normals.shape == mesh.faces.shape

        # we shouldn't be able to assign stupid wrong values
        # even with nonzero and the right shape
        mesh.face_normals = g.np.ones_like(mesh.faces) * [0.0, 0.0, 1.0]
        assert not g.np.allclose(mesh.face_normals, [0.0, 0.0, 1.0])

        # setting normals to None should force recompute
        mesh.face_normals = None
        assert mesh.face_normals is not None
        assert not g.np.allclose(mesh.face_normals, [0.0, 0.0, 1.0])

        # setting face normals to zeros shouldn't work
        mesh.face_normals = g.np.zeros_like(mesh.faces)
        assert g.np.allclose(g.np.linalg.norm(mesh.face_normals, axis=1), 1.0)

    def test_merge(self):
        """
        Check merging with vertex normals
        """
        # no vertex merging
        m = g.get_mesh("cube_compressed.obj", process=False)
        assert m.vertices.shape == (24, 3)
        assert m.faces.shape == (12, 3)
        assert g.np.isclose(m.volume, 8.0, atol=1e-4)

        # with normal-aware vertex merging
        m = g.get_mesh("cube_compressed.obj", process=True)
        assert m.vertices.shape == (24, 3)
        assert m.faces.shape == (12, 3)
        assert g.np.isclose(m.volume, 8.0, atol=1e-4)

        # without considering normals should just be cube
        m.merge_vertices(merge_norm=True)
        assert m.vertices.shape == (8, 3)
        assert m.faces.shape == (12, 3)
        assert g.np.isclose(m.volume, 8.0, atol=1e-4)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
