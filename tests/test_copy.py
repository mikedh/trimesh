"""
Copy meshes and make sure they do what we expect.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class CopyTests(g.unittest.TestCase):
    def test_copy(self):
        for mesh in g.get_meshes(raise_error=True):
            if not isinstance(mesh, g.trimesh.Trimesh) or len(mesh.faces) == 0:
                continue
            start = hash(mesh)

            # make sure some stuff is populated
            _ = mesh.kdtree
            _ = mesh.triangles_tree
            _ = mesh.face_adjacency_angles
            _ = mesh.facets
            assert "triangles_tree" in mesh._cache
            assert len(mesh._cache) > 0

            # if you cache c-objects then deepcopy the mesh
            # it randomly segfaults
            copy_count = 200
            for _i in range(copy_count):
                copied = mesh.copy(include_cache=False)
                assert len(copied._cache) == 0
                assert len(mesh._cache) > 0

                # deepcopy should clear the cache
                copied = g.copy.deepcopy(mesh)
                assert len(copied._cache) == 0
                assert len(mesh._cache) > 0

                # regular copy should try to preserve the cache
                copied = g.copy.copy(mesh)
                assert len(copied._cache) == len(mesh._cache)
                # the triangles_tree should be the SAME OBJECT
                assert id(copied.triangles_tree) == id(mesh.triangles_tree)

                # cache should be same data in different object
                assert id(copied._cache.cache) != id(mesh._cache.cache)
                assert id(copied._cache) != id(mesh._cache)
                # identifier shouldn't change
                assert g.np.allclose(copied.identifier, mesh.identifier)

            # ...still shouldn't have changed anything
            assert start == hash(mesh)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
