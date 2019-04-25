try:
    from . import generic as g
except BaseException:
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
        # mesh with a single tetrahedron
        tet = g.get_mesh('tet.ply')

        for engine in self.engines:
            # without requiring watertight the split should be into every face
            split = soup.split(only_watertight=False, engine=engine)
            assert len(split) == len(soup.faces)

            # with watertight there should be an empty list
            split = soup.split(only_watertight=True, engine=engine)
            assert len(split) == 0

            split = mult.split(only_watertight=False, engine=engine)
            assert len(split) >= 119

            split = mult.split(only_watertight=True, engine=engine)
            assert len(split) >= 117

            # random triangles should have no facets
            facets = g.trimesh.graph.facets(mesh=soup, engine=engine)
            assert len(facets) == 0

            facets = g.trimesh.graph.facets(mesh=mult, engine=engine)
            assert all(len(i) >= 2 for i in facets)
            assert len(facets) >= 8654

            split = sing.split(only_watertight=False, engine=engine)
            assert len(split) == 1
            assert split[0].is_watertight
            assert split[0].is_winding_consistent

            split = sing.split(only_watertight=True, engine=engine)
            assert len(split) == 1
            assert split[0].is_watertight
            assert split[0].is_winding_consistent

            # single tetrahedron
            assert tet.is_volume
            assert tet.body_count == 1
            # regardless of method or flag we should have one body result
            split = tet.split(only_watertight=True, engine=engine)
            assert len(split) == 1
            split = tet.split(only_watertight=False, engine=engine)
            assert len(split) == 1

    def test_vertex_adjacency_graph(self):
        f = g.trimesh.graph.vertex_adjacency_graph

        # a mesh with a single watertight body
        sing = g.get_mesh('featuretype.STL')
        vert_adj_g = f(sing)
        assert len(sing.vertices) == len(vert_adj_g)

    def test_engine_time(self):
        for mesh in g.get_meshes():
            tic = [g.time.time()]
            for engine in self.engines:
                split = mesh.split(engine=engine, only_watertight=False)  # NOQA
                facets = g.trimesh.graph.facets(mesh=mesh, engine=engine)  # NOQA
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

    def test_watertight(self):
        m = g.get_mesh('shared.STL')  # NOQA
        # assert m.is_watertight
        # assert m.is_winding_consistent
        # assert m.is_volume

    def test_traversals(self):
        """
        Test traversals (BFS+DFS)
        """

        # generate some simple test data
        simple_nodes = g.np.arange(20)
        simple_edges = g.np.column_stack((simple_nodes[:-1],
                                          simple_nodes[1:]))
        simple_edges = g.np.vstack((
            simple_edges,
            [[19, 0],
             [10, 1000],
             [500, 501]])).astype(g.np.int64)

        all_edges = g.data['edges']
        all_edges.append(simple_edges)

        for edges in all_edges:
            edges = g.np.array(edges, dtype=g.np.int64)
            assert g.trimesh.util.is_shape(edges, (-1, 2))

            # collect the new nodes
            nodes = g.np.unique(edges)

            # the basic BFS/DFS traversal
            dfs_basic = g.trimesh.graph.traversals(edges, 'dfs')
            bfs_basic = g.trimesh.graph.traversals(edges, 'bfs')
            # check return types
            assert all(i.dtype == g.np.int64 for i in dfs_basic)
            assert all(i.dtype == g.np.int64 for i in bfs_basic)

            # check to make sure traversals visited every node
            dfs_set = set(g.np.hstack(dfs_basic))
            bfs_set = set(g.np.hstack(bfs_basic))
            nodes_set = set(nodes)
            assert dfs_set == nodes_set
            assert bfs_set == nodes_set

            # check traversal filling
            # fill_traversals should always include every edge
            # regardless of the path so test on bfs/dfs/empty
            for traversal in [dfs_basic, bfs_basic, []]:
                # disconnect consecutive nodes that are not edges
                # and add edges that were left off by jumps
                dfs = g.trimesh.graph.fill_traversals(traversal, edges)
                # edges that are included in the new separated traversal
                inc = g.trimesh.util.vstack_empty(
                    [g.np.column_stack((i[:-1], i[1:]))
                     for i in dfs])

                # make a set from edges included in the traversal
                inc_set = set(g.trimesh.grouping.hashable_rows(
                    g.np.sort(inc, axis=1)))
                # make a set of the source edges we were supposed to include
                edge_set = set(g.trimesh.grouping.hashable_rows(
                    g.np.sort(edges, axis=1)))

                # we should have exactly the same edges
                # after the filled traversal as we started with
                assert len(inc) == len(edges)
                # every edge should occur exactly once
                assert len(inc_set) == len(inc)
                # unique edges should be the same
                assert inc_set == edge_set

                # check all return dtypes
                assert all(i.dtype == g.np.int64 for i in dfs)


def check_engines(edges, nodes):
    """
    Make sure connected component graph engines are
    returning the exact same values
    """
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
