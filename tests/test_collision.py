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


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
