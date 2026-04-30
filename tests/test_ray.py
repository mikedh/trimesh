try:
    from . import generic as g
except BaseException:
    import generic as g


def test_rays():
    meshes = [g.get_mesh(**k) for k in g.data["ray_data"]["load_kwargs"]]
    rays = g.data["ray_data"]["rays"]
    names = [m.source.file_name for m in meshes]

    hit_id = []
    hit_loc = []
    hit_any = []
    for m in meshes:
        name = m.source.file_name
        hit_any.append(m.ray.intersects_any(**rays[name]))
        hit_loc.append(m.ray.intersects_location(**rays[name])[0])
        hit_id.append(m.ray.intersects_id(**rays[name]))
    hit_any = g.np.array(hit_any, dtype=g.np.int64)

    for i in g.trimesh.grouping.group(g.np.unique(names, return_inverse=True)[1]):
        broken = g.np.ptp(hit_any[i].astype(g.np.int64), axis=0).sum()
        assert broken == 0


def test_rps():
    from trimesh import ray as ray_mod

    for use_embree in [True, False] if ray_mod.has_embree else [False]:
        dimension = (10000, 3)
        sphere = g.get_mesh("unit_sphere.STL")

        # blow up the number of triangles so we actually get a test
        sphere = sphere.subdivide().subdivide().subdivide()

        # assign a different ray intersector
        if use_embree:
            sphere.ray = ray_mod.ray_pyembree.RayMeshIntersector(sphere)
        else:
            # create a ray-mesh query object for the current mesh
            # initializing is very inexpensive and object is convenient to have.
            # On first query expensive bookkeeping is done (creation of r-tree),
            # and is cached for subsequent queries
            sphere.ray = ray_mod.ray_triangle.RayMeshIntersector(sphere)

        ray_origins = g.random(dimension)
        ray_directions = g.np.tile([0, 0, 1], (dimension[0], 1))
        ray_origins[:, 2] = -5

        # force ray object to allocate tree before timing it
        # tree = sphere.ray.tree
        tic = [g.time.time()]
        a = sphere.ray.intersects_id(ray_origins, ray_directions)
        tic.append(g.time.time())
        b = sphere.ray.intersects_location(ray_origins, ray_directions)
        tic.append(g.time.time())

        # make sure ray functions always return numpy arrays
        assert all(len(i.shape) >= 0 for i in a)
        assert all(len(i.shape) >= 0 for i in b)

        rps = dimension[0] / g.np.diff(tic)

        g.log.info(
            f"Measured {rps} rays/second on {len(sphere.faces)} triangles with embree={use_embree}"
        )


def test_empty():
    """
    Test queries with no hits
    """
    for use_embree in [True, False]:
        dimension = (100, 3)
        sphere = g.get_mesh("unit_sphere.STL", use_embree=use_embree)
        # should never hit the sphere
        ray_origins = g.random(dimension)
        ray_directions = g.np.tile([0, 1, 0], (dimension[0], 1))
        ray_origins[:, 2] = -5

        # make sure ray functions always return numpy arrays
        # these functions return multiple results all of which
        # should always be a numpy array
        assert all(
            len(i.shape) >= 0
            for i in sphere.ray.intersects_id(ray_origins, ray_directions)
        )
        assert all(
            len(i.shape) >= 0
            for i in sphere.ray.intersects_location(ray_origins, ray_directions)
        )


def test_contains():
    scale = 1.5
    for use_embree in [True, False]:
        mesh = g.get_mesh("unit_cube.STL", use_embree=use_embree)
        g.log.info("Contains test ray engine: " + str(mesh.ray.__class__))

        test_on = mesh.ray.contains_points(mesh.vertices)  # NOQA
        test_in = mesh.ray.contains_points(mesh.vertices * (1.0 / scale))
        assert test_in.all()

        test_out = mesh.ray.contains_points(mesh.vertices * scale)
        assert not test_out.any()

        points_way_out = (g.random((30, 3)) * 100) + 1.0 + mesh.bounds[1]
        test_way_out = mesh.ray.contains_points(points_way_out)
        assert not test_way_out.any()

        test_centroid = mesh.ray.contains_points([mesh.center_mass])
        assert test_centroid.all()


def test_contains_cavity():
    # https://github.com/mikedh/trimesh/issues/2534
    from trimesh import ray as ray_mod

    mesh = g.trimesh.boolean.difference(
        [
            g.trimesh.creation.box(extents=[1, 1, 1]),
            g.trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
        ]
    )
    engines = [ray_mod.ray_triangle.RayMeshIntersector]
    if ray_mod.has_embree:
        engines.append(ray_mod.ray_pyembree.RayMeshIntersector)
    for engine in engines:
        mesh.ray = engine(mesh)
        # origin sits inside the subtracted cavity
        assert not mesh.contains([[0, 0, 0]])[0]


def test_on_vertex():
    for use_embree in [False]:
        m = g.trimesh.creation.box(use_embree=use_embree)

        origins = g.np.zeros_like(m.vertices)
        vectors = m.vertices.copy()

        assert m.ray.intersects_any(ray_origins=origins, ray_directions=vectors).all()

        (_locations, index_ray, _index_tri) = m.ray.intersects_location(
            ray_origins=origins, ray_directions=vectors
        )

        hit_count = g.np.bincount(index_ray, minlength=len(origins))

        assert (hit_count == 1).all()


def test_on_edge():
    for use_embree in [True, False]:
        m = g.get_mesh("7_8ths_cube.stl", use_embree=use_embree)

        points = [[4.5, 0, -23], [4.5, 0, -2], [0, -1e-4, 0.0], [0, 0, -1]]
        truth = [False, True, True, True]
        result = g.trimesh.ray.ray_util.contains_points(m.ray, points)

        assert (result == truth).all()


def test_multiple_hits():
    """ """
    # Set camera focal length (in pixels)
    f = g.np.array([1000.0, 1000.0])
    h, w = 256, 256

    # Set up a list of ray directions - one for each pixel in our (256,
    # 256) output image.
    ray_directions = g.trimesh.util.grid_arange(
        [[-h / 2, -w / 2], [h / 2, w / 2]], step=2.0
    )
    ray_directions = g.np.column_stack(
        (ray_directions, g.np.ones(len(ray_directions)) * f[0])
    )

    # Initialize the camera origin to be somewhere behind the cube.
    cam_t = g.np.array([0, 0, -15.0])
    # Duplicate to ensure we have an camera_origin per ray direction
    ray_origins = g.np.tile(cam_t, (ray_directions.shape[0], 1))

    for use_embree in [True, False]:
        # Generate a 1 x 1 x 1 cube using the trimesh box primitive
        cube_mesh = g.trimesh.creation.box(extents=[2, 2, 2], use_embree=use_embree)

        # Perform 256 * 256 raycasts, one for each pixel on the image
        # plane. We only want the 'first' hit.
        index_triangles, _index_ray = cube_mesh.ray.intersects_id(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False,
        )
        assert len(g.np.unique(index_triangles)) == 2

        index_triangles, _index_ray = cube_mesh.ray.intersects_id(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True
        )
        assert len(g.np.unique(index_triangles)) > 2


def test_contain_single():
    # not watertight
    mesh = g.get_mesh("teapot.stl", use_embree=False)

    # sample a grid of points (n,3)
    points = mesh.bounding_box.sample_grid(step=2.0)
    # to a contains check on every point
    contained = mesh.ray.contains_points(points)

    assert len(points) == len(contained)

    # not contained and should surface a bug
    for point in mesh.bounding_box.vertices:
        mesh.ray.contains_points([point])


def test_box():
    """
    Run box- ray intersection along Z and make sure XY match
    ray origin XY.
    """

    for kwargs in [{"use_embree": True}, {"use_embree": False}]:
        mesh = g.get_mesh("unit_cube.STL", **kwargs)
        # grid is across meshes XY profile
        origins = g.trimesh.util.grid_linspace(
            mesh.bounds[:, :2] + g.np.reshape([-0.02, 0.02], (-1, 1)), 100
        )
        origins = g.np.column_stack((origins, g.np.ones(len(origins)) * -100))
        # all vectors are along Z axis
        vectors = g.np.ones((len(origins), 3)) * [0, 0, 1.0]

        # (n,3) float intersection position in space
        # (n,) int, index of original ray
        # (m,) int, index of mesh.faces
        pos, ray, _tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=vectors
        )

        for p, r in zip(pos, ray):
            # intersect location XY should match ray origin XY
            assert g.np.allclose(p[:2], origins[r][:2])
            # the Z of the hit should be on the cube's
            # top or bottom face
            assert g.np.isclose(p[2], mesh.bounds[:, 2]).any()


def test_embree_duplicate_hits():
    """
    Test that multiple_hits=True does not get stuck in a float32 precision
    loop when coordinates are extremely large
    """
    try:
        from embreex import __version__
    except ImportError:
        g.log.warning("no embree to test")
        return

    if int(__version__.split(".", 1)[0]) < 4:
        g.log.warning(f"only testing duplicate hits on embreex>4: got `{__version__}`")
        return
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    for scale in [1e-4, 0.1, 1.0, 10, 1e4, 1e8]:
        # create box at different scales to test
        # our behavior with floating-point precision bug
        mesh = g.trimesh.creation.box().subdivide().subdivide()
        mesh.apply_scale(scale)

        # make sure this is using embree
        # and disable unit scaling to test numeric behavior
        # scale_to_box will make MOST floating point things go away
        # but the logic in the intersector, particularly for multi-hit
        # requires "exponential offset" when f32 gets "stuck" on a triangle
        # so our tests here should be correct even at small or large scales
        mesh.ray = RayMeshIntersector(mesh, scale_to_box=False)

        # Fire a ray straight from exactly on the box surface
        ray_origins = g.np.array([[0.0, 0.0, scale / 2.0]])
        ray_directions = g.np.array([[0.0, 0.0, -1.0]])

        # do the ray query
        index_tri, _ = mesh.ray.intersects_id(
            ray_origins, ray_directions, multiple_hits=True
        )

        # The ray should hit exactly two faces through the cube
        assert len(index_tri) == 2, (scale, index_tri)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    # test_on_edge()
    # test_rps()
    test_embree_duplicate_hits()
