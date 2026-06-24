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

        # Test one-to-many collision checking
        m = g.trimesh.collision.CollisionManager()
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

    def test_ignored_pairs_internal(self):
        # Feature added in https://github.com/mikedh/trimesh/issues/2454
        #
        # The reporter wants to skip designed contacts between articulated
        # robot links so the manager only flags unintended collisions.
        # Build a 4-cube manager where every pair overlaps then progressively
        # ignore pairs and verify both the `names` set AND the boolean-only
        # result drop the right pairs (the bool path goes through a separate
        # fast-path that also has to respect the ignore set).
        if fcl is None:
            return
        cube = g.get_mesh("unit_cube.STL")

        m = g.trimesh.collision.CollisionManager()
        m.add_object("a", cube)
        # all three of the others sit just inside the unit cube `a` so
        # they all collide with each other and with `a`
        for name, dx in [("b", 0.3), ("c", 0.6), ("d", -0.3)]:
            tf = g.np.eye(4)
            tf[:3, 3] = [dx, 0, 0]
            m.add_object(name, cube, tf)

        # sanity: with no ignored pairs the full clique shows up
        result, names = m.in_collision_internal(return_names=True)
        assert result is True
        assert ("a", "b") in names
        # bool-only path
        assert m.in_collision_internal() is True

        # ignore one pair and confirm it is the ONLY one removed and the
        # rest are reported untouched
        m.set_pair_ignored("a", "b")
        assert m.ignored_pairs == {("a", "b")}
        assert m.is_pair_ignored("a", "b")
        # symmetric look-up
        assert m.is_pair_ignored("b", "a")
        result, names = m.in_collision_internal(return_names=True)
        assert result is True
        assert ("a", "b") not in names
        assert ("a", "c") in names
        # bool-only path must NOT short-circuit on the ignored pair
        assert m.in_collision_internal() is True

        # ignore every remaining pair → both the names and the bool
        # result must drop to "no collision"
        all_pairs = [
            ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d"),
        ]
        for x, y in all_pairs:
            m.set_pair_ignored(x, y)
        result, names = m.in_collision_internal(return_names=True)
        assert result is False
        assert len(names) == 0
        assert m.in_collision_internal() is False

        # turning ignored=False re-enables the pair
        m.set_pair_ignored("a", "b", ignored=False)
        assert not m.is_pair_ignored("a", "b")
        result, names = m.in_collision_internal(return_names=True)
        assert result is True
        assert ("a", "b") in names

    def test_ignored_pairs_input_validation(self):
        # Guard rails on the new API: ignoring a name against itself or
        # referencing a name not in the manager must raise immediately
        # — silent no-ops would let user typos hide real collisions.
        if fcl is None:
            return
        cube = g.get_mesh("unit_cube.STL")
        m = g.trimesh.collision.CollisionManager()
        m.add_object("link0", cube)
        # separate `link1` from `link0` so the post-clear assertion
        # checks the genuine "no collision" path
        far = g.np.eye(4)
        far[:3, 3] = [10.0, 0, 0]
        m.add_object("link1", cube, far)

        try:
            m.set_pair_ignored("link0", "link0")
        except ValueError:
            pass
        else:
            raise AssertionError("self-ignore should raise")

        try:
            m.set_pair_ignored("link0", "ghost")
        except ValueError:
            pass
        else:
            raise AssertionError("missing-name should raise")

        # is_pair_ignored never raises and is direction-independent
        assert not m.is_pair_ignored("link0", "link1")
        m.set_pair_ignored("link0", "link1")
        assert m.is_pair_ignored("link0", "link1")
        assert m.is_pair_ignored("link1", "link0")

        # clear_ignored_pairs wipes the lot
        m.clear_ignored_pairs()
        assert m.ignored_pairs == set()
        assert m.in_collision_internal() is False  # boxes overlap-free at origin pair

    def test_ignored_pairs_cleared_on_remove(self):
        # remove_object must also evict any ignored-pair entries
        # involving the removed name so a future re-add of the same
        # name doesn't silently inherit stale ignore rules.
        if fcl is None:
            return
        cube = g.get_mesh("unit_cube.STL")
        m = g.trimesh.collision.CollisionManager()
        m.add_object("a", cube)
        tf = g.np.eye(4)
        tf[:3, 3] = [0.3, 0, 0]
        m.add_object("b", cube, tf)
        m.add_object("c", cube, tf)
        m.set_pair_ignored("a", "b")
        m.set_pair_ignored("a", "c")
        assert m.ignored_pairs == {("a", "b"), ("a", "c")}

        m.remove_object("a")
        # both pairs referenced "a" so the ignore set must be empty
        assert m.ignored_pairs == set()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
