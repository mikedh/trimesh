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

    def test_scene(self):
        if fcl is None:
            return
        scene = g.get_mesh("cycloidal.3DXML")
        manager, objects = g.trimesh.collision.scene_to_collision(scene)

        assert manager.in_collision_internal()
        assert objects is not None


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
