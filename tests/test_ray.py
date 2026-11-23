try:
    from . import generic as g
except BaseException:
    import generic as g


def test_engines_against_truth():
    # rays through a placed box against its analytic slab crossings and
    # through an icosphere against its inradius and circumradius: every
    # hit count, and every hit on its face's plane at the true distance
    random = g.np.random.default_rng(11)
    extents = g.np.array([2.0, 3.0, 4.0])
    transform = g.trimesh.transformations.random_rotation_matrix(random.random(3))
    transform[:3, 3] = [1.0, -2.0, 0.5]
    box = g.trimesh.creation.box(extents=extents, transform=transform)
    inverse = g.np.linalg.inv(transform)

    # from outside toward points inside the box, so each crosses twice,
    # from inside in every direction, crossing once, and away from it
    targets = g.trimesh.transform_points(
        (random.random((500, 3)) - 0.5) * extents * 0.9, transform
    )
    outside = box.centroid + g.trimesh.util.unitize(random.normal(size=(500, 3))) * 20
    inside = g.trimesh.transform_points(
        (random.random((500, 3)) - 0.5) * extents * 0.9, transform
    )
    origins = g.np.vstack((outside, inside, outside))
    directions = g.trimesh.util.unitize(
        g.np.vstack(
            (targets - outside, random.normal(size=(500, 3)), outside - box.centroid)
        )
    )
    # the slab crossings in the box's own frame
    local_origins = g.trimesh.transform_points(origins, inverse)
    local_directions = directions @ inverse[:3, :3].T
    with g.np.errstate(divide="ignore"):
        low = (-extents / 2.0 - local_origins) / local_directions
        high = (extents / 2.0 - local_origins) / local_directions
    enter = g.np.minimum(low, high).max(axis=1)
    leave = g.np.maximum(low, high).min(axis=1)
    expected = [2] * 500 + [1] * 500 + [0] * 500
    lengths = random.uniform(0.5, 3.0, (len(origins), 1))

    sphere = g.trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    inradius = g.np.abs((sphere.triangles[:, 0] * sphere.face_normals).sum(axis=1)).min()
    sphere_origins = g.trimesh.util.unitize(random.normal(size=(1000, 3))) * 10.0
    sphere_targets = g.trimesh.util.unitize(random.normal(size=(1000, 3))) * (
        random.random((1000, 1)) * 4.0
    )
    sphere_directions = g.trimesh.util.unitize(sphere_targets - sphere_origins)
    # how close each ray passes to the center
    approach = g.np.linalg.norm(g.np.cross(sphere_directions, -sphere_origins), axis=1)

    # an STL whose stored face normals are off by degrees
    feature = g.get_mesh("featuretype.STL")
    feature_origins = feature.centroid + random.normal(size=(1000, 3)) * feature.scale
    # each aimed through a triangle's centroid so it hits at least once
    feature_directions = (
        feature.triangles_center[random.integers(0, len(feature.faces), 1000)]
        - feature_origins
    )
    for engine, precision in g.ray_engines():
        box.ray = engine(box)
        # directions of any length, hits in world units
        triangles, rays, locations = box.ray.intersects_id(
            origins, directions * lengths, multiple_hits=True, return_locations=True
        )
        assert g.np.bincount(rays, minlength=len(origins)).tolist() == expected
        # every hit at one of its ray's slab crossings, and together a
        # ray's hits are exactly its crossings ahead of the origin
        distances = ((locations - origins[rays]) * directions[rays]).sum(axis=1)
        nearest = g.np.minimum(
            g.np.abs(distances - enter[rays]), g.np.abs(distances - leave[rays])
        )
        assert (nearest < precision * 10).all()
        sums = g.np.bincount(rays, weights=distances, minlength=len(origins))
        assert g.np.allclose(sums[:500], (enter + leave)[:500], atol=precision * 20)
        assert g.np.allclose(sums[500:1000], leave[500:1000], atol=precision * 10)
        # the first hit of every ray that hits, and -1 for the rest
        first = box.ray.intersects_first(origins, directions)
        assert ((first >= 0) == (g.np.array(expected) > 0)).all()
        # on the plane of the face it reports
        offsets = locations - box.triangles[triangles, 0]
        planes = (offsets * box.face_normals[triangles]).sum(axis=1)
        assert g.np.allclose(planes, 0.0, atol=precision * 10)

        sphere.ray = engine(sphere)
        _, sphere_rays = sphere.ray.intersects_id(
            sphere_origins, sphere_directions, multiple_hits=True
        )
        counts = g.np.bincount(sphere_rays, minlength=len(sphere_origins))
        assert (counts[approach < inradius * 0.999] == 2).all()
        assert (counts[approach > 2.0 * 1.001] == 0).all()
        assert (
            sphere.ray.intersects_any(sphere_origins, sphere_directions) == (counts > 0)
        ).all()

        # every hit on the triangle it reports, which an engine trusting
        # the stored normals misses: embree's fix is its own change
        feature.ray = engine(feature)
        if "pyembree" in type(feature.ray).__module__:
            continue
        triangles, rays, locations = feature.ray.intersects_id(
            feature_origins, feature_directions, return_locations=True
        )
        assert (g.np.bincount(rays, minlength=1000) > 0).all()
        closest = g.trimesh.triangles.closest_point(
            feature.triangles[triangles], locations
        )
        gaps = g.np.linalg.norm(closest - locations, axis=1)
        assert (gaps < precision * feature.scale).all()


def test_hierarchy_brute_force():
    # the numpy engine finds exactly the hits of every ray against every
    # triangle, for soups, a teapot in a stadium, giant and flat triangles,
    # rays along an axis, with a zero component and from inside, once
    # each, in batches of any size whose memory is bounded per batch
    import tracemalloc

    from trimesh.ray.ray_triangle import bounding_volume_hierarchy, ray_triangle_id

    random = g.np.random.default_rng(4)
    soup = random.random((300, 3, 3))
    stadium = g.np.vstack(
        (random.random((500, 3, 3)) * 1e-3, random.random((5, 3, 3)) * 100.0)
    )
    giant = g.np.vstack((random.random((200, 3, 3)), random.random((3, 3, 3)) * 50 - 25))
    flat = random.random((300, 3, 3))
    flat[:, :, 2] = 0.5
    for triangles in [soup, stadium, giant, flat, soup[:1]]:
        lower, upper = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
        origins = lower - (upper - lower) + random.random((1000, 3)) * (upper - lower) * 3
        directions = random.normal(size=(1000, 3))
        directions[:150] = g.np.eye(3)[random.integers(0, 3, 150)]
        directions[150:300, random.integers(0, 3)] = 0.0
        origins[300:450] = lower + random.random((150, 3)) * (upper - lower)

        # brute force: each ray against each triangle's plane, kept where
        # the crossing is ahead of the origin and inside the triangle
        ray, triangle = (
            index.ravel()
            for index in g.np.meshgrid(
                g.np.arange(len(origins)), g.np.arange(len(triangles)), indexing="ij"
            )
        )
        normals = g.trimesh.unitize(
            g.np.cross(*(triangles[:, 1:] - triangles[:, :1]).swapaxes(0, 1))
        )
        locations, valid = g.trimesh.intersections.planes_lines(
            triangles[triangle, 0], normals[triangle], origins[ray], directions[ray]
        )
        ray, triangle = ray[valid], triangle[valid]
        barycentric = g.trimesh.triangles.points_to_barycentric(
            triangles[triangle], locations
        )
        ahead = ((locations - origins[ray]) * directions[ray]).sum(axis=1)
        inside = ((barycentric > -g.tol.zero) & (barycentric < 1 + g.tol.zero)).all(
            axis=1
        ) & (ahead > -g.tol.zero)
        truth = set(zip(triangle[inside].tolist(), ray[inside].tolist()))

        hierarchy = bounding_volume_hierarchy(triangles)
        for batch_size in [2**14, 997]:
            found, rays, _ = ray_triangle_id(
                triangles,
                origins,
                directions,
                hierarchy=hierarchy,
                batch_size=batch_size,
            )
            assert len(found) == len(truth)
            assert set(zip(found.tolist(), rays.tolist())) == truth

    # the hierarchy scales with the triangles, padding included,
    # and an empty one is never entered
    order, levels = bounding_volume_hierarchy(soup)
    scaled = bounding_volume_hierarchy(soup * 1e-8)
    assert g.np.array_equal(order, scaled[0])
    assert all(
        g.np.allclose(level * 1e-8, small, rtol=1e-9, atol=0.0, equal_nan=True)
        for level, small in zip(levels, scaled[1])
    )
    assert len(ray_triangle_id(soup, origins, directions)[0]) > 0
    assert all(
        len(column) == 0 for column in ray_triangle_id(soup[:0], origins, directions)
    )

    # sixteen batches peak at the memory of about one
    sphere = g.trimesh.creation.icosphere(subdivisions=4)
    hierarchy = bounding_volume_hierarchy(sphere.triangles)
    origins = g.trimesh.unitize(random.normal(size=(2**14, 3))) * 3.0
    directions = random.normal(size=origins.shape) * 0.3 - origins
    peaks = []
    for count in [2**10, 2**14]:
        tracemalloc.start()
        ray_triangle_id(
            sphere.triangles,
            origins[:count],
            directions[:count],
            hierarchy=hierarchy,
            batch_size=2**10,
        )
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    assert peaks[1] < peaks[0] * 1.5


def test_scene_intersect():
    # boxes placed at known rotations against their analytic slab
    # crossings, with the node of every hit, through trimkit's own API
    if not g.trimesh.util.has_module("trimkit"):
        return
    from trimkit import RayScene

    random = g.np.random.default_rng(9)
    extents = g.np.array([1.0, 2.0, 3.0])
    box = g.trimesh.creation.box(extents=extents)
    scene = g.trimesh.Scene()
    # a node with no faces to hit comes first so positions can't line up
    scene.add_geometry(
        g.trimesh.load_path([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), node_name="line"
    )
    scene.add_geometry(box, geom_name="box", node_name="node_0")
    transforms = {}
    for index in range(4):
        transform = g.trimesh.transformations.random_rotation_matrix(random.random(3))
        transform[:3, 3] = [index * 20.0, 0.0, 0.0]
        transforms[f"node_{index}"] = transform
        scene.graph.update(frame_to=f"node_{index}", matrix=transform, geometry="box")

    # from above each box toward points inside it, crossing it twice
    aimed = g.np.repeat(list(transforms), 100)
    targets = g.np.vstack(
        [
            g.trimesh.transform_points(
                (random.random((100, 3)) - 0.5) * extents * 0.9, transforms[node]
            )
            for node in transforms
        ]
    )
    origins = targets + [0.0, 0.0, 8.0] + random.normal(size=targets.shape) * 0.5
    directions = (targets - origins) * random.uniform(0.5, 3.0, (len(origins), 1))
    unit = g.trimesh.util.unitize(directions)
    # each ray's slab crossings in the frame of the box it's aimed at
    inverses = g.np.array([g.np.linalg.inv(transforms[node]) for node in aimed])
    local_origins = (inverses[:, :3, :3] @ origins[:, :, None])[:, :, 0] + inverses[
        :, :3, 3
    ]
    local_directions = (inverses[:, :3, :3] @ unit[:, :, None])[:, :, 0]
    low = (-extents / 2.0 - local_origins) / local_directions
    high = (extents / 2.0 - local_origins) / local_directions
    enter = g.np.minimum(low, high).max(axis=1)
    leave = g.np.maximum(low, high).min(axis=1)

    ray = scene.ray
    assert isinstance(ray, RayScene)
    rays, instances, faces, distances = ray.intersect(origins, directions)
    assert g.np.bincount(rays, minlength=len(origins)).tolist() == [2] * len(origins)
    # each hit is on the node it was aimed at, on a face of that node's
    # own mesh, and the two hits of a ray are its slab crossings
    assert (ray.nodes[instances] == aimed[rays]).all()
    locations = origins[rays] + unit[rays] * distances[:, None]
    local = (inverses[rays, :3, :3] @ locations[:, :, None])[:, :, 0] + inverses[
        rays, :3, 3
    ]
    closest = g.trimesh.triangles.closest_point(box.triangles[faces], local)
    assert g.np.allclose(closest, local, atol=1e-6)
    sums = g.np.bincount(rays, weights=distances, minlength=len(origins))
    assert g.np.allclose(sums, enter + leave, atol=1e-6)

    # each ray's first hit is where it enters its own box
    rays, instances, _, distances = ray.intersect(origins, directions, maximum_hits=1)
    assert rays.tolist() == list(range(len(origins)))
    assert (ray.nodes[instances] == aimed).all()
    assert g.np.allclose(distances, enter, atol=1e-6)
    assert ray.intersect_any(origins, directions).all()
    assert not ray.intersect_any(origins, -directions).any()
    # the world box of every node, the line's empty
    assert len(ray.boxes()) == len(ray.nodes)

    # the scene's query takes no instances: other instances, like many
    # poses, are trimkit's own call on the meshes already built
    alone = g.np.eye(4).reshape((1, 4, 4))
    with g.pytest.raises(TypeError):
        ray.intersect(origins, directions, alone, g.np.ones(1, dtype=g.np.int64))
    assert RayScene.intersect_any(
        ray,
        origins[:100],
        directions[:100] + [20.0, 0.0, 0.0],
        alone,
        g.np.ones(1, dtype=g.np.int64),
    ).shape == (100,)

    # it pickles as itself with its scene, and the scene pickles with it built
    restored = g.pickle.loads(g.pickle.dumps(ray))
    assert type(restored) is type(ray)
    assert restored.intersect_any(origins, directions).all()
    assert (
        g.pickle.loads(g.pickle.dumps(scene)).ray.intersect_any(origins, directions).all()
    )

    # moving a node in a replaced graph moves its hits but never rebuilds
    # the meshes it casts at
    scene.graph = scene.graph.copy()
    moved = transforms["node_0"].copy()
    moved[:3, 3] += [0.0, 50.0, 0.0]
    scene.graph.update(frame_to="node_0", matrix=moved)
    assert not ray.intersect_any(origins[:100], directions[:100]).any()
    assert ray.intersect_any(origins[100:], directions[100:]).all()
    assert scene.ray is ray
    # and changing geometry does
    box.apply_scale(2.0)
    assert scene.ray is not ray


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
    # rays per second for every engine on its first query, which builds
    # its acceleration structure, and its second, which must return the
    # same hits, by orders of magnitude of triangles and an assembly
    models = {
        f"icosphere {subdivisions}": g.trimesh.creation.icosphere(
            subdivisions=subdivisions
        )
        for subdivisions in [3, 5]
    }
    cycloidal = g.get_mesh("cycloidal.3DXML")
    for factor in [0.0, 0.1, 1.0]:
        exploded = cycloidal.copy()
        exploded.explode(factor)
        models[f"cycloidal {factor:.0%}"] = exploded.to_mesh()

    random = g.np.random.default_rng(7)
    report = [f"{'model':>16}{'triangles':>10}{'engine':>30}{'cold':>12}{'warm':>12}"]
    for name, mesh in models.items():
        # from outside the bounds toward points inside them
        targets = mesh.bounds[0] + random.random((10000, 3)) * mesh.extents
        origins = mesh.centroid + mesh.scale * g.trimesh.unitize(
            random.normal(size=targets.shape)
        )
        for engine, _ in g.ray_engines():
            mesh.ray = engine(mesh)
            tic = [g.time.perf_counter()]
            cold = mesh.ray.intersects_id(origins, targets - origins)
            tic.append(g.time.perf_counter())
            warm = mesh.ray.intersects_id(origins, targets - origins)
            tic.append(g.time.perf_counter())
            assert len(cold[0]) > 0
            assert all(g.np.array_equal(*pair) for pair in zip(cold, warm))
            rps = "".join(f"{r:>12.0f}" for r in len(origins) / g.np.diff(tic))
            report.append(
                f"{name:>16}{len(mesh.faces):>10}{type(mesh.ray).__module__:>30}{rps}"
            )
    g.log.info("rays per second:\n" + "\n".join(report))


def test_empty():
    """
    Test queries with no hits
    """
    sphere = g.get_mesh("unit_sphere.STL")
    for engine, _ in g.ray_engines():
        sphere.ray = engine(sphere)
        dimension = (100, 3)
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
    mesh = g.get_mesh("unit_cube.STL")
    for engine, _ in g.ray_engines():
        mesh.ray = engine(mesh)

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
    mesh = g.trimesh.boolean.difference(
        [
            g.trimesh.creation.box(extents=[1, 1, 1]),
            g.trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
        ]
    )
    for engine, _precision in g.ray_engines():
        mesh.ray = engine(mesh)
        # origin sits inside the subtracted cavity
        assert not mesh.contains([[0, 0, 0]])[0]


def test_on_vertex():
    m = g.trimesh.creation.box()
    for engine, _ in g.ray_engines():
        m.ray = engine(m)

        origins = g.np.zeros_like(m.vertices)
        vectors = m.vertices.copy()

        assert m.ray.intersects_any(ray_origins=origins, ray_directions=vectors).all()

        (_locations, index_ray, _index_tri) = m.ray.intersects_location(
            ray_origins=origins, ray_directions=vectors
        )

        hit_count = g.np.bincount(index_ray, minlength=len(origins))

        assert (hit_count == 1).all()


def test_on_edge():
    m = g.get_mesh("7_8ths_cube.stl")
    for engine, _ in g.ray_engines():
        m.ray = engine(m)

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

    # Generate a 2 x 2 x 2 cube using the trimesh box primitive
    cube_mesh = g.trimesh.creation.box(extents=[2, 2, 2])
    for engine, _ in g.ray_engines():
        cube_mesh.ray = engine(cube_mesh)

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

    mesh = g.get_mesh("unit_cube.STL")
    for engine, _ in g.ray_engines():
        mesh.ray = engine(mesh)
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
    test_rps()
