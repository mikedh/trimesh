try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import fcl
except BaseException:
    fcl = None


class CollisionTest(g.unittest.TestCase):
    def test_collision(self):
        # Ensure that FCL is importable
        if fcl is None:
            g.log.warning("skipping FCL tests: not installed")
            return

        cube = g.get_mesh("unit_cube.STL")

        tf1 = g.np.eye(4)
        tf1[:3, 3] = g.np.array([5, 0, 0])

        tf2 = g.np.eye(4)
        tf2[:3, 3] = g.np.array([-5, 0, 0])

        # Test one-to-many collision checking, deprecated with a warning
        # only where its replacement `Scene.collision` works
        with g.warnings.catch_warnings(record=True) as caught:
            g.warnings.simplefilter("always")
            m = g.trimesh.collision.CollisionManager()
        deprecated = [w for w in caught if "Scene.collision" in str(w.message)]
        assert len(deprecated) == int(g.trimesh.util.has_module("trimkit"))
        assert all(w.category is DeprecationWarning for w in deprecated)
        m.add_object("cube0", cube)
        m.add_object("cube1", cube, tf1)

        ret = m.in_collision_single(cube)
        assert ret is True

        ret, names, data = m.in_collision_single(
            cube, tf1, return_names=True, return_data=True
        )

        assert ret is True
        for c in data:
            assert g.np.allclose(c.point, g.np.array([5.0, -0.5, 0.5]))
            assert g.np.isclose(c.depth, 1.0)
            assert g.np.allclose(c.normal, g.np.array([-1.0, 0.0, 0.0]))

        if "cube1" not in names:
            g.log.debug("\n\n", m._objs.keys(), names)
        assert "cube1" in names

        ret, names, data = m.in_collision_single(
            cube, tf2, return_names=True, return_data=True
        )
        assert ret is False
        assert len(names) == 0
        assert all(len(i.point) == 3 for i in data)

        # Test internal collision checking and object
        # addition/removal/modification
        ret = m.in_collision_internal()
        assert ret is False

        m.add_object("cube2", cube, tf1)
        ret, names = m.in_collision_internal(return_names=True)
        assert ret is True
        assert ("cube1", "cube2") in names
        assert ("cube0", "cube1") not in names
        assert ("cube2", "cube1") not in names

        m.set_transform("cube2", tf2)
        ret = m.in_collision_internal()
        assert ret is False

        m.set_transform("cube2", tf1)
        ret = m.in_collision_internal()
        assert ret is True

        m.remove_object("cube2")
        ret = m.in_collision_internal()
        assert ret is False

        # Test manager-to-manager collision checking
        m = g.trimesh.collision.CollisionManager()
        m.add_object("cube0", cube)
        m.add_object("cube1", cube, tf1)

        n = g.trimesh.collision.CollisionManager()
        n.add_object("cube0", cube, tf2)

        ret = m.in_collision_other(n)
        assert ret is False

        n.add_object("cube3", cube, tf1)

        ret = m.in_collision_other(n)
        assert ret is True

        ret, names = m.in_collision_other(n, return_names=True)
        assert ret is True
        assert ("cube1", "cube3") in names
        assert ("cube3", "cube1") not in names

    def test_random_spheres(self):
        if fcl is None:
            g.log.warning("skipping FCL tests: not installed")
            return

        # check to see if a scene with a bunch of random
        # spheres
        spheres = [
            g.trimesh.creation.icosphere(radius=i[0]).apply_translation(i[1:] * 100)
            for i in g.random((1000, 4))
        ]
        scene = g.trimesh.Scene(spheres)
        manager, _ = g.trimesh.collision.scene_to_collision(scene)
        collides = manager.in_collision_internal()
        assert isinstance(collides, bool)

    def test_distance(self):
        if fcl is None:
            g.log.warning("skipping FCL tests: not installed")
            return

        cube = g.get_mesh("unit_cube.STL")

        tf1 = g.np.eye(4)
        tf1[:3, 3] = g.np.array([5, 0, 0])

        tf2 = g.np.eye(4)
        tf2[:3, 3] = g.np.array([-5, 0, 0])

        tf3 = g.np.eye(4)
        tf3[:3, 3] = g.np.array([2, 0, 0])

        tf4 = g.np.eye(4)
        tf4[:3, 3] = g.np.array([-2, 0, 0])

        tf5 = g.np.eye(4)
        tf5[:3, 3] = g.np.array([5.75, 0, 0])

        # Test one-to-many distance checking
        m = g.trimesh.collision.CollisionManager()
        m.add_object("cube1", cube, tf1)

        dist = m.min_distance_single(cube)
        assert g.np.isclose(dist, 4.0)

        dist = m.min_distance_single(cube, tf5)
        assert g.np.isclose(dist, -0.25)

        dist, name = m.min_distance_single(cube, return_name=True)
        assert g.np.isclose(dist, 4.0)
        assert name == "cube1"

        m.add_object("cube2", cube, tf2)

        dist, name = m.min_distance_single(cube, tf3, return_name=True)
        assert g.np.isclose(dist, 2.0)
        assert name == "cube1"

        dist, name = m.min_distance_single(cube, tf4, return_name=True)
        assert g.np.isclose(dist, 2.0)
        assert name == "cube2"

        # Test internal distance checking and object
        # addition/removal/modification
        dist = m.min_distance_internal()
        assert g.np.isclose(dist, 9.0)

        dist, names = m.min_distance_internal(return_names=True)
        assert g.np.isclose(dist, 9.0)
        assert names == ("cube1", "cube2")

        m.add_object("cube3", cube, tf3)

        dist, names = m.min_distance_internal(return_names=True)
        assert g.np.isclose(dist, 2.0)
        assert names == ("cube1", "cube3")

        m.set_transform("cube3", tf4)

        dist, names = m.min_distance_internal(return_names=True)
        assert g.np.isclose(dist, 2.0)
        assert names == ("cube2", "cube3")

        dist, names = m.min_distance_internal(name="cube1", return_names=True)
        assert g.np.isclose(dist, 6.0)
        assert names == ("cube1", "cube3")

        # Test manager-to-manager distance checking
        m = g.trimesh.collision.CollisionManager()
        m.add_object("cube0", cube)
        m.add_object("cube1", cube, tf1)

        n = g.trimesh.collision.CollisionManager()
        n.add_object("cube4", cube, tf2)

        dist, names, data = m.min_distance_other(n, return_names=True, return_data=True)
        assert g.np.isclose(dist, 4.0)
        assert names == ("cube0", "cube4")
        assert g.np.isclose(
            g.np.linalg.norm(data.point(names[0]) - data.point(names[1])), dist
        )

        n.add_object("cube5", cube, tf4)

        dist, names, data = m.min_distance_other(n, return_names=True, return_data=True)
        assert g.np.isclose(dist, 1.0)
        assert names == ("cube0", "cube5")
        assert g.np.isclose(
            g.np.linalg.norm(data.point(names[0]) - data.point(names[1])), dist
        )

    def test_collision_batch(self):
        # regression: in_collision_single(return_names=True) dropped pairs
        # once cumulative contacts hit num_max_contacts — fcl's default
        # callback halts traversal at the cap
        if fcl is None:
            g.log.warning("skipping FCL tests: not installed")
            return

        # many probes each colliding with one large static object
        anvil = g.trimesh.creation.box(extents=[200, 200, 200])
        probe = g.trimesh.creation.revolve(
            g.np.array([[0, 3.0], [3, 3], [3, 75], [127, 75], [127, 580], [0, 580]]),
            sections=32,
        )
        count = 700
        names = [f"probe_{i}" for i in range(count)]
        batch = g.trimesh.collision.CollisionManager()
        for i, name in enumerate(names):
            t = g.np.eye(4)
            t[:3, 3] = [(i / count - 0.5) * 30, 0, -50]
            batch.add_object(name, probe, transform=t)

        hit, reported = batch.in_collision_single(anvil, return_names=True)
        assert hit
        assert set(names) == reported

    def test_scene(self):
        if fcl is None:
            return
        scene = g.get_mesh("cycloidal.3DXML")
        manager, objects = g.trimesh.collision.scene_to_collision(scene)

        assert manager.in_collision_internal()
        assert objects is not None


def test_scene_collides():
    # unit cubes placed at known offsets against the offsets themselves:
    # which touch, how deep, how far apart, and where, from one cached
    # trimkit scene that moving a node never rebuilds
    if not g.trimesh.util.has_module("trimkit"):
        return
    offsets = {
        "first": [0.0, 0.0, 0.0],
        "second": [0.75, 0.0, 0.0],
        "third": [3.0, 0.0, 0.0],
        "fourth": [0.0, 6.0, 0.0],
    }
    cube = g.trimesh.creation.box()
    scene = g.trimesh.Scene()
    scene.add_geometry(cube, geom_name="cube", node_name="first")
    for node, offset in offsets.items():
        scene.graph.update(
            frame_to=node,
            matrix=g.trimesh.transformations.translation_matrix(offset),
            geometry="cube",
        )

    def truth(first, second):
        # the signed distance between two axis aligned unit cubes
        separation = g.np.abs(g.np.subtract(offsets[first], offsets[second]))
        gaps = g.np.maximum(separation - 1.0, 0.0)
        if (gaps > 0).any():
            return g.np.linalg.norm(gaps)
        return -(1.0 - separation).min()

    from trimkit import CollisionScene

    # the scene's trimkit scene, which every trimkit query works on
    collision = scene.collision
    assert isinstance(collision, CollisionScene)
    nodes = collision.nodes
    lookup = {node: instance for instance, node in enumerate(nodes)}

    result = collision.collides()
    assert nodes[result.pairs].tolist() == [["first", "second"]]
    assert result.faces is None and result.distances is None

    # the contact: as deep as the overlap, along an axis, inside both cubes
    result = collision.collides(return_distance=True)
    assert nodes[result.pairs].tolist() == [["first", "second"]]
    assert g.np.allclose(result.distances, -0.25)
    assert g.np.allclose(g.np.abs(result.normals).max(axis=1), 1.0)
    assert (result.faces == -1).all()
    for pair, points in zip(nodes[result.pairs], result.points):
        for node, point in zip(pair, points):
            assert (g.np.abs(point - offsets[node]) <= 0.5 + g.tol.merge).all()

    # every pair's signed distance, and the closest points that far apart
    result = collision.distances()
    assert len(result) == 6
    assert g.np.allclose(result.distances, [truth(*pair) for pair in nodes[result.pairs]])
    apart = result.distances > 0
    assert g.np.allclose(
        g.np.linalg.norm(result.points[apart, 1] - result.points[apart, 0], axis=1),
        result.distances[apart],
    )

    # named pairs only, in the order given
    pairs = g.np.array([[lookup["third"], lookup["first"]]], dtype=g.np.int64)
    result = collision.distances(pairs=pairs)
    assert nodes[result.pairs].tolist() == [["third", "first"]]
    assert g.np.allclose(result.distances, truth("third", "first"))
    assert len(collision.collides(pairs=pairs)) == 0

    # the scene's query takes no instances: other instances, like many
    # poses, are trimkit's own call on the meshes already built
    alone = g.np.eye(4).reshape((1, 4, 4))
    single = g.np.zeros(1, dtype=g.np.int64)
    with g.pytest.raises(TypeError):
        collision.collides(alone, single)
    assert len(CollisionScene.collides(collision, alone, single)) == 0

    # it pickles as itself with its scene, and the scene pickles with it built
    restored = g.pickle.loads(g.pickle.dumps(collision))
    assert type(restored) is type(collision)
    assert restored.nodes[restored.collides().pairs].tolist() == [["first", "second"]]
    reloaded = g.pickle.loads(g.pickle.dumps(scene)).collision
    assert reloaded.nodes[reloaded.collides().pairs].tolist() == [["first", "second"]]

    # anything that isn't a mesh is an instance nothing touches
    stale = collision
    scene.add_geometry(
        g.trimesh.load_path([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        geom_name="line",
        node_name="line",
    )
    # one built before the geometry changed refuses to guess its instances
    with g.pytest.raises(KeyError):
        stale.collides()
    collision = scene.collision
    assert collision is not stale
    nodes = collision.nodes
    assert nodes[collision.collides().pairs].tolist() == [["first", "second"]]
    result = collision.distances()
    line = (nodes[result.pairs] == "line").any(axis=1)
    assert line.sum() == 4 and g.np.isinf(result.distances[line]).all()
    assert g.np.allclose(
        result.distances[~line], [truth(*pair) for pair in nodes[result.pairs][~line]]
    )

    # moving a node in a replaced graph changes the answer but never
    # rebuilds the trimkit scene, which reads the scene's current graph
    scene.graph = scene.graph.copy()
    offsets["third"] = [1.5, 0.0, 0.0]
    scene.graph.update(
        frame_to="third",
        matrix=g.trimesh.transformations.translation_matrix(offsets["third"]),
    )
    assert scene.collision is collision
    assert nodes[collision.collides().pairs].tolist() == [
        ["first", "second"],
        ["second", "third"],
    ]
    # and changing geometry does
    cube.apply_scale(2.0)
    assert scene.collision is not collision

    # meshes that aren't convex touch on real faces
    machine = g.get_mesh("cycloidal.3DXML")
    collision = machine.collision
    result = collision.collides(return_distance=True)
    assert len(result) > 0
    _, index = machine.graph.instance_matrices(list(machine.geometry))
    meshes = list(machine.geometry.values())
    for pair, faces in zip(result.pairs, result.faces):
        for instance, face in zip(pair, faces):
            assert -1 <= face < len(meshes[index[instance]].faces)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
