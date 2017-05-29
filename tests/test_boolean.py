import generic as g


class BooleanTest(g.unittest.TestCase):

    def setUp(self):
        self.a = g.get_mesh('ballA.off')
        self.b = g.get_mesh('ballB.off')
        self.truth = g.data['boolean']

    def is_zero(self, value):
        return abs(value) < .001

    def test_boolean(self):
        a, b = self.a, self.b

        engines = [('blender', g.trimesh.interfaces.blender.exists),
                   ('scad',    g.trimesh.interfaces.scad.exists)]

        for engine, exists in engines:
            if not exists:
                g.log.warning('skipping boolean engine %s', engine)
                continue

            g.log.info('Testing boolean ops with engine %s', engine)
            d = a.difference(b, engine=engine)
            if not d.is_watertight:
                d.show()
            self.assertTrue(d.is_watertight)
            self.assertTrue(self.is_zero(d.volume - self.truth['difference']))

            i = a.intersection(b, engine=engine)
            self.assertTrue(i.is_watertight)
            self.assertTrue(self.is_zero(
                i.volume - self.truth['intersection']))

            u = a.union(b, engine=engine)
            self.assertTrue(u.is_watertight)
            self.assertTrue(self.is_zero(u.volume - self.truth['union']))

            g.log.info('booleans succeeded with %s', engine)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
