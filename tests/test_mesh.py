"""
Load all the meshes we can get our hands on and check things, stuff.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g


def test_meshes():
    # make sure we can load everything we think we can
    formats = g.trimesh.available_formats()
    assert all(isinstance(i, str) for i in formats)
    assert all(len(i) > 0 for i in formats)
    assert all(i in formats for i in ["stl", "ply", "off", "obj", "glb"])

    for mesh in g.get_meshes(raise_error=True):
        # log file name for debugging
        file_name = mesh.source.file_name

        # ply files can return PointCloud objects
        if file_name.startswith("points_"):
            continue

        g.log.info("Testing %s", file_name)
        start = {mesh.__hash__(), mesh.__hash__()}
        assert len(mesh.faces) > 0
        assert len(mesh.vertices) > 0

        # make sure vertex normals match vertices and are valid
        assert mesh.vertex_normals.shape == mesh.vertices.shape
        assert g.np.isfinite(mesh.vertex_normals).all()

        # should be one per vertex
        assert len(mesh.vertex_faces) == len(mesh.vertices)

        # check some edge properties
        assert len(mesh.edges) > 0
        assert len(mesh.edges_unique) > 0
        assert len(mesh.edges_sorted) == len(mesh.edges)
        assert len(mesh.edges_face) == len(mesh.edges)

        # check edges_unique
        assert len(mesh.edges) == len(mesh.edges_unique_inverse)
        assert g.np.allclose(
            mesh.edges_sorted, mesh.edges_unique[mesh.edges_unique_inverse]
        )
        assert len(mesh.edges_unique) == len(mesh.edges_unique_length)

        # euler number should be an integer
        assert isinstance(mesh.euler_number, int)

        # check bounding primitives
        assert mesh.bounding_box.volume > 0.0
        assert mesh.bounding_primitive.volume > 0.0

        # none of these should have mutated anything
        assert start == {mesh.__hash__(), mesh.__hash__()}

        # run processing, again
        mesh.process()

        # still shouldn't have changed anything
        assert start == {mesh.__hash__(), mesh.__hash__()}

        if not (mesh.is_watertight and mesh.is_winding_consistent):
            continue

        assert len(mesh.facets) == len(mesh.facets_area)
        assert len(mesh.facets) == len(mesh.facets_normal)
        assert len(mesh.facets) == len(mesh.facets_boundary)

        if len(mesh.facets) != 0:
            faces = mesh.facets[mesh.facets_area.argmax()]
            outline = mesh.outline(faces)
            # check to make sure we can generate closed paths
            # on a Path3D object
            test = outline.paths  # NOQA

        smoothed = mesh.smooth_shaded  # NOQA

        assert abs(mesh.volume) > 0.0

        mesh.section(plane_normal=[0, 0, 1], plane_origin=mesh.centroid)

        sample = mesh.sample(1000)
        even_sample = g.trimesh.sample.sample_surface_even(mesh, 100)  # NOQA
        assert sample.shape == (1000, 3)
        g.log.info("finished testing meshes")

        # make sure vertex kdtree and triangles rtree exist

        t = mesh.kdtree
        assert hasattr(t, "query")
        g.log.info("Creating triangles tree")
        r = mesh.triangles_tree
        assert hasattr(r, "intersection")
        g.log.info("Triangles tree ok")

        # face angles should have same
        assert mesh.face_angles.shape == mesh.faces.shape
        assert len(mesh.vertices) == len(mesh.vertex_defects)
        assert len(mesh.principal_inertia_components) == 3

        # make a ray query which may lead to unpicklable caching
        dimension = (100, 3)
        ray_origins = g.random(dimension)
        ray_directions = g.np.tile([0, 0, 1], (dimension[0], 1))
        ray_origins[:, 2] = mesh.bounds[0][2] - mesh.scale

        # call additional C objects
        assert mesh.kdtree is not None
        assert mesh.triangles_tree is not None

        # force ray object to be created
        ray = mesh.ray.intersects_location(ray_origins, ray_directions)
        assert ray is not None

        # collect list of cached properties that are writeable
        writeable = []

        # make sure a roundtrip pickle works
        # if the cache has non-pickleable stuff this will break
        pickle = g.pickle.dumps(mesh)
        assert isinstance(pickle, bytes)
        assert len(pickle) > 0

        r = g.pickle.loads(pickle)
        assert r.faces.shape == mesh.faces.shape
        assert g.np.isclose(r.volume, mesh.volume)

        # we should have built up a bunch of stuff into
        # our cache, so make sure all numpy arrays cached
        # are read-only and not crazy
        for name, cached in mesh._cache.cache.items():
            # only check numpy arrays
            if not isinstance(cached, g.np.ndarray):
                continue

            # nothing in the cache should be writeable
            if cached.flags["WRITEABLE"]:
                raise ValueError(f"{name} is writeable!")

            # only check int, float, and bool
            if cached.dtype.kind not in "ibf":
                continue

            # there should never be NaN values
            if g.np.isnan(cached).any():
                raise ValueError("NaN values in %s/%s", file_name, name)

            # fields allowed to have infinite values
            if name in ["face_adjacency_radius"]:
                continue

            # make sure everything is finite
            if not g.np.isfinite(cached).all():
                raise ValueError("inf values in %s/%s", file_name, name)

        # ...still shouldn't have changed anything
        assert start == {mesh.__hash__(), mesh.__hash__()}

        # log the names of properties we need to make read-only
        if len(writeable) > 0:
            # TODO : all cached values should be read-only
            g.log.error("cached properties writeable: {}".format(", ".join(writeable)))


def test_mesh_2D():
    # check a simple mesh with 2D vertices
    m = g.trimesh.Trimesh(
        vertices=g.np.random.random((100, 2)),
        faces=g.np.arange(99, dtype=g.np.int64).reshape((-1, 3)),
    )
    # the face normals should be 3D (+Z)
    assert m.face_normals.shape == m.faces.shape

    rend = g.trimesh.rendering.mesh_to_vertexlist(m)
    assert len(rend) > 1


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_mesh_2D()
