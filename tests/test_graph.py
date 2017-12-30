import generic as g


class GraphTest(g.unittest.TestCase):

    def setUp(self):
        self.engines = ['scipy', 'networkx']
        if g.trimesh.graph._has_gt:
            self.engines.append('graphtool')
        else:
            g.log.warning('No graph-tool to test!')

    def test_soup(self):
        # a soup of random triangles, with no adjacent pairs
        soup = g.get_mesh('soup.stl')

        assert len(soup.face_adjacency) == 0
        assert len(soup.face_adjacency_radius) == 0
        assert len(soup.face_adjacency_edges) == 0
        assert len(soup.face_adjacency_convex) == 0
        assert len(soup.face_adjacency_unshared) == 0
        assert len(soup.face_adjacency_angles) == 0
        assert len(soup.facets) == 0

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

            # random triangles should have no facets
            facets = g.trimesh.graph.facets(mesh=soup, engine=engine)
            self.assertTrue(len(facets) == 0)

            facets = g.trimesh.graph.facets(mesh=mult, engine=engine)
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

    def test_vertex_adjacency_graph(self):
        f = g.trimesh.graph.vertex_adjacency_graph

        # a mesh with a single watertight body
        sing = g.get_mesh('featuretype.STL')
        vert_adj_g = f(sing)
        self.assertTrue(len(sing.vertices) == len(vert_adj_g))

    def test_engine_time(self):
        for mesh in g.get_meshes():
            tic = [g.time.time()]
            for engine in self.engines:
                split = mesh.split(engine=engine, only_watertight=False)
                facets = g.trimesh.graph.facets(mesh=mesh, engine=engine)
                tic.append(g.time.time())

            tic_diff = g.np.diff(tic)
            tic_min = tic_diff.min()
            tic_diff /= tic_min
            g.log.info('graph engine on %s (scale %f sec):\n%s',
                       mesh.metadata['file_name'],
                       tic_min,
                       str(g.np.column_stack((self.engines,
                                              tic_diff))))

    def test_smoothed(self):
        mesh = g.get_mesh('ADIS16480.STL')
        assert len(mesh.faces) == len(mesh.smoothed().faces)

    def test_engines(self):
        edges = g.np.arange(10).reshape((-1, 2))
        for i in range(0, 20):
            check_engines(nodes=g.np.arange(i),
                          edges=edges)
        edges = g.np.column_stack((g.np.arange(1, 11),
                                   g.np.arange(0, 10)))
        for i in range(0, 20):
            check_engines(nodes=g.np.arange(i),
                          edges=edges)


def check_engines(edges, nodes):
    '''
    Make sure connected component graph engines are
    returning the exact same values
    '''
    results = []
    engines = [None, 'scipy', 'networkx']

    for engine in engines:
        c = g.trimesh.graph.connected_components(edges,
                                                 nodes=nodes,
                                                 engine=engine)
        if len(c) > 0:
            # check to see if every resulting component was in the
            # set of nodes
            diff = g.np.setdiff1d(g.np.hstack(c), nodes)
            assert len(diff) == 0
        results.append(
            sorted(
                g.trimesh.util.md5_object(
                    g.np.sort(i)) for i in c))
    assert all(i == results[0] for i in results)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
