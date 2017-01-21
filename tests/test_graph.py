import generic as g


class GraphTest(g.unittest.TestCase):

    def setUp(self):
        self.engines = ['scipy', 'networkx']
        if g.trimesh.graph._has_gt:
            self.engines.append('graphtool')
        else:
            g.log.warning('No graph-tool to test!')

    def test_components(self):
        # a soup of random triangles, with no adjacent pairs
        soup = g.get_mesh('soup.stl')
        # a mesh with multiple watertight bodies
        mult = g.get_mesh('cycloidal.ply')
        # a mesh with a single watertight body
        sing = g.get_mesh('featuretype.STL')

        for engine in self.engines:
            # without requiring watertight the split should be into every face
            split = soup.split(only_watertight=False, engine=engine)
            self.assertTrue(len(split) == len(soup.faces))

            # with watertight there should be an empty list
            split = soup.split(only_watertight=True, engine=engine)
            self.assertTrue(len(split) == 0)

            split = mult.split(only_watertight=False, engine=engine)
            self.assertTrue(len(split) >= 119)

            split = mult.split(only_watertight=True, engine=engine)
            self.assertTrue(len(split) >= 117)

            facets = soup.facets(engine=engine)
            self.assertTrue(len(facets) == 0)

            facets = mult.facets(engine=engine)
            self.assertTrue(all(len(i) >= 2 for i in facets))
            self.assertTrue(len(facets) >= 8654)

            split = sing.split(only_watertight=False, engine=engine)
            self.assertTrue(len(split) == 1)
            self.assertTrue(split[0].is_watertight)
            self.assertTrue(split[0].is_winding_consistent)

            split = sing.split(only_watertight=True, engine=engine)
            self.assertTrue(len(split) == 1)
            self.assertTrue(split[0].is_watertight)
            self.assertTrue(split[0].is_winding_consistent)

    def test_engine_time(self):
        for mesh in g.get_meshes():
            tic = [g.time.time()]
            for engine in self.engines:
                split = mesh.split(engine=engine, only_watertight=False)
                facets = mesh.facets(engine=engine)
                tic.append(g.time.time())

            tic_diff = g.np.diff(tic)
            tic_min = tic_diff.min()
            tic_diff /= tic_min
            g.log.info('graph engine on %s (scale %f sec):\n%s',
                       mesh.metadata['file_name'],
                       tic_min,
                       str(g.np.column_stack((self.engines,
                                              tic_diff))))

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
