import generic as g


class CollisionTest(g.unittest.TestCase):

    def test_collision(self):
        # Ensure that FCL is importable
        try:
            g.trimesh.collision.CollisionManager()
        except ValueError:
            g.log.warning('skipping collision tests, no FCL installed')
            return

        cube = g.get_mesh('unit_cube.STL')

        tf1 = g.np.eye(4)
        tf1[:3, 3] = g.np.array([5, 0, 0])

        tf2 = g.np.eye(4)
        tf2[:3, 3] = g.np.array([-5, 0, 0])

        # Test one-to-many collision checking
        m = g.trimesh.collision.CollisionManager()
        m.add_object('cube0', cube)
        m.add_object('cube1', cube, tf1)

        ret = m.in_collision_single(cube)
        self.assertTrue(ret == True)

        ret, names = m.in_collision_single(cube, tf1, return_names=True)
        self.assertTrue(ret == True)
        
        if 'cube1' not in names:
            print('\n\n', m._objs.keys(), names)
        self.assertTrue('cube1' in names)

        ret, names = m.in_collision_single(cube, tf2, return_names=True)
        self.assertTrue(ret == False)
        self.assertTrue(len(names) == 0)

        # Test internal collision checking and object
        # addition/removal/modification
        ret = m.in_collision_internal()
        self.assertTrue(ret == False)

        m.add_object('cube2', cube, tf1)
        ret, names = m.in_collision_internal(return_names=True)
        self.assertTrue(ret == True)
        self.assertTrue(('cube1', 'cube2') in names)
        self.assertTrue(('cube0', 'cube1') not in names)
        self.assertTrue(('cube2', 'cube1') not in names)

        m.set_transform('cube2', tf2)
        ret = m.in_collision_internal()
        self.assertTrue(ret == False)

        m.set_transform('cube2', tf1)
        ret = m.in_collision_internal()
        self.assertTrue(ret == True)

        m.remove_object('cube2')
        ret = m.in_collision_internal()
        self.assertTrue(ret == False)

        # Test manager-to-manager collision checking
        m = g.trimesh.collision.CollisionManager()
        m.add_object('cube0', cube)
        m.add_object('cube1', cube, tf1)

        n = g.trimesh.collision.CollisionManager()
        n.add_object('cube0', cube, tf2)

        ret = m.in_collision_other(n)
        self.assertTrue(ret == False)

        n.add_object('cube3', cube, tf1)

        ret = m.in_collision_other(n)
        self.assertTrue(ret == True)

        ret, names = m.in_collision_other(n, return_names=True)
        self.assertTrue(ret == True)
        self.assertTrue(('cube1', 'cube3') in names)
        self.assertTrue(('cube3', 'cube1') not in names)

    def test_distance(self):
        # Ensure that FCL is importable
        try:
            g.trimesh.collision.CollisionManager()
        except ValueError:
            g.log.warning('skipping collision tests, no FCL installed')
            return

        cube = g.get_mesh('unit_cube.STL')

        tf1 = g.np.eye(4)
        tf1[:3, 3] = g.np.array([5, 0, 0])

        tf2 = g.np.eye(4)
        tf2[:3, 3] = g.np.array([-5, 0, 0])

        tf3 = g.np.eye(4)
        tf3[:3, 3] = g.np.array([2, 0, 0])

        tf4 = g.np.eye(4)
        tf4[:3, 3] = g.np.array([-2, 0, 0])

        # Test one-to-many distance checking
        m = g.trimesh.collision.CollisionManager()
        m.add_object('cube1', cube, tf1)

        dist = m.min_distance_single(cube)
        self.assertTrue(g.np.isclose(dist, 4.0))

        dist, name = m.min_distance_single(cube, return_name=True)
        self.assertTrue(g.np.isclose(dist, 4.0))
        self.assertTrue(name == 'cube1')

        m.add_object('cube2', cube, tf2)

        dist, name = m.min_distance_single(cube, tf3, return_name=True)
        self.assertTrue(g.np.isclose(dist, 2.0))
        self.assertTrue(name == 'cube1')

        dist, name = m.min_distance_single(cube, tf4, return_name=True)
        self.assertTrue(g.np.isclose(dist, 2.0))
        self.assertTrue(name == 'cube2')

        # Test internal distance checking and object
        # addition/removal/modification
        dist = m.min_distance_internal()
        self.assertTrue(g.np.isclose(dist, 9.0))

        dist, names = m.min_distance_internal(return_names=True)
        self.assertTrue(g.np.isclose(dist, 9.0))
        self.assertTrue(names == ('cube1', 'cube2'))

        m.add_object('cube3', cube, tf3)

        dist, names = m.min_distance_internal(return_names=True)
        self.assertTrue(g.np.isclose(dist, 2.0))
        self.assertTrue(names == ('cube1', 'cube3'))

        m.set_transform('cube3', tf4)

        dist, names = m.min_distance_internal(return_names=True)
        self.assertTrue(g.np.isclose(dist, 2.0))
        self.assertTrue(names == ('cube2', 'cube3'))

        # Test manager-to-manager distance checking
        m = g.trimesh.collision.CollisionManager()
        m.add_object('cube0', cube)
        m.add_object('cube1', cube, tf1)

        n = g.trimesh.collision.CollisionManager()
        n.add_object('cube0', cube, tf2)

        dist, names = m.min_distance_other(n, return_names=True)
        self.assertTrue(g.np.isclose(dist, 4.0))
        self.assertTrue(names == ('cube0', 'cube0'))

        n.add_object('cube4', cube, tf4)

        dist, names = m.min_distance_other(n, return_names=True)
        self.assertTrue(g.np.isclose(dist, 1.0))
        self.assertTrue(names == ('cube0', 'cube4'))


    def test_scene(self):
        try:
            import fcl
        except ImportError:
            return
        scene = g.get_mesh('cycloidal.3DXML')
        
        manager, objects = g.trimesh.collision.scene_to_collision(scene)

        assert manager.in_collision_internal()

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
