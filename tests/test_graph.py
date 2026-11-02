try:
    from . import generic as g
except BaseException:
    import generic as g


def _box(extents, invert=False):
    mesh = g.trimesh.creation.box(extents=extents)
    if invert:
        mesh.invert()
    return mesh


class GraphTest(g.unittest.TestCase):
    def setUp(self):
        self.engines = []
        try:
            self.engines.append("scipy")
        except BaseException:
            pass
        try:
            self.engines.append("networkx")
        except BaseException:
            pass

    def test_soup(self):
        # a soup of random triangles, with no adjacent pairs
        soup = g.get_mesh("soup.stl")

        assert len(soup.face_adjacency) == 0
        assert len(soup.face_adjacency_radius) == 0
        assert len(soup.face_adjacency_edges) == 0
        assert len(soup.face_adjacency_convex) == 0
        assert len(soup.face_adjacency_unshared) == 0
        assert len(soup.face_adjacency_angles) == 0
        assert len(soup.facets) == 0

    def test_components(self):
        # a soup of random triangles, with no adjacent pairs
        soup = g.get_mesh("soup.stl")
        # a mesh with multiple watertight bodies
        mult = g.get_mesh("cycloidal.ply")
        # a mesh with a single watertight body
        sing = g.get_mesh("featuretype.STL")
        # mesh with a single tetrahedron
        tet = g.get_mesh("tet.ply")

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

    def test_split_solids(self):
        # grouping of closed shells into solids by containment parity
        for engine in self.engines:
            # legacy split is purely topological: cavity shell is separate
            hollow = _box([2, 2, 2]) + _box([1, 1, 1], invert=True)
            assert len(hollow.split(engine=engine)) == 2

            split = hollow.split(solids=True, engine=engine)
            assert len(split) == 1
            assert split[0].is_watertight
            assert split[0].is_winding_consistent
            assert g.np.isclose(split[0].volume, 7.0)

            # a solid nested inside a cavity is a separate material region
            nested = _box([6, 6, 6]) + _box([4, 4, 4], invert=True) + _box([2, 2, 2])
            split = nested.split(solids=True, engine=engine)
            assert len(split) == 2
            assert g.np.allclose(g.np.sort([i.volume for i in split]), [8.0, 152.0])
            assert all(i.is_watertight for i in split)

            # nesting deeper than one cavity: solid-cavity-solid-cavity
            # exercises immediate-parent lookup at nesting degree >= 2
            deep = (
                _box([8, 8, 8])
                + _box([6, 6, 6], invert=True)
                + _box([4, 4, 4])
                + _box([2, 2, 2], invert=True)
            )
            split = deep.split(solids=True, engine=engine)
            assert len(split) == 2
            assert all(i.is_watertight for i in split)
            # outer solid is 8**3 - 6**3, inner solid is 4**3 - 2**3
            assert g.np.allclose(g.np.sort([i.volume for i in split]), [56.0, 296.0])

            # disjoint solids: grouping is a no-op, matches legacy split
            disjoint = _box([1, 1, 1]) + _box([1, 1, 1]).apply_translation([5, 0, 0])
            assert len(disjoint.split(solids=True, engine=engine)) == len(
                disjoint.split(engine=engine)
            )

    def test_split_solids_orient(self):
        # orientation normalization of grouped shells
        for engine in self.engines:
            hollow = _box([2, 2, 2]) + _box([1, 1, 1], invert=True)

            # orient=True on already correct input is a no-op: faces bit-identical
            a = hollow.split(solids=True, orient=True, engine=engine)[0]
            b = hollow.split(solids=True, orient=False, engine=engine)[0]
            assert g.np.array_equal(a.faces, b.faces)

            # outward-wound cavity: orient flips it to a real cavity (7.0),
            # without orient the raw input winding is preserved (9.0)
            outward_cavity = _box([2, 2, 2]) + _box([1, 1, 1])
            split = outward_cavity.split(solids=True, orient=False, engine=engine)
            assert len(split) == 1
            assert g.np.isclose(split[0].volume, 9.0)
            split = outward_cavity.split(solids=True, orient=True, engine=engine)
            assert len(split) == 1
            assert g.np.isclose(split[0].volume, 7.0)

            # orient is decoupled from repair: with orient=False the winding is
            # identical whether or not repair runs
            a = hollow.split(solids=True, repair=True, orient=False, engine=engine)[0]
            b = hollow.split(solids=True, repair=False, orient=False, engine=engine)[0]
            assert g.np.array_equal(a.faces, b.faces)

            # fully raw path: repair=False, orient=False alters nothing on
            # already-correct closed input
            raw = hollow.split(solids=True, repair=False, orient=False, engine=engine)
            assert len(raw) == 1
            assert raw[0].is_watertight
            assert g.np.isclose(raw[0].volume, 7.0)

            # a single inverted shell must still be oriented: orient does not
            # depend on how many other bodies happen to be present
            single = _box([2, 2, 2], invert=True)
            split = single.split(solids=True, orient=True, engine=engine)
            assert len(split) == 1
            assert g.np.isclose(split[0].volume, 8.0)
            assert split[0].is_winding_consistent
            # without orient the input winding is preserved
            split = single.split(solids=True, orient=False, engine=engine)
            assert g.np.isclose(split[0].volume, -8.0)

            # a watertight but winding-inconsistent shell (one face reversed so
            # every edge still appears exactly twice) still classifies as closed
            base = _box([2, 2, 2])
            messy_faces = base.faces.copy()
            messy_faces[0] = messy_faces[0][::-1]
            messy_shell = g.trimesh.Trimesh(
                vertices=base.vertices.copy(), faces=messy_faces, process=False
            )
            assert messy_shell.is_watertight
            assert not messy_shell.is_winding_consistent
            messy = messy_shell + _box([1, 1, 1], invert=True)
            # legacy topological split still returns both watertight shells
            assert len(messy.split(engine=engine)) == 2
            # the cavity is grouped into the inconsistent outer shell, no error
            split = messy.split(solids=True, engine=engine)
            assert len(split) == 1
            assert len(split[0].faces) == 24
            # orient cannot derive a direction for the inconsistent shell, so
            # the result is identical whether or not orientation is requested
            a = messy.split(solids=True, orient=True, engine=engine)[0]
            b = messy.split(solids=True, orient=False, engine=engine)[0]
            assert g.np.array_equal(a.faces, b.faces)

    def test_split_solids_filtering(self):
        # interaction with only_watertight and min_faces filtering
        for engine in self.engines:
            hollow = _box([2, 2, 2]) + _box([1, 1, 1], invert=True)

            # open components are not attached to a solid; existing filtering applies
            loose = g.trimesh.Trimesh(
                vertices=[[5, 0, 0], [6, 0, 0], [5, 1, 0]],
                faces=[[0, 1, 2]],
                process=False,
            )
            with_open = hollow + loose
            assert (
                len(with_open.split(solids=True, only_watertight=False, engine=engine))
                == 2
            )
            assert (
                len(with_open.split(solids=True, only_watertight=True, engine=engine))
                == 1
            )

            # min_faces drops a solid group without corrupting the winding of
            # the groups that remain (regression: flips must stay aligned)
            small = _box([0.5, 0.5, 0.5], invert=True).apply_translation([10, 0, 0])
            mixed = small + hollow
            split = mixed.split(solids=True, orient=True, min_faces=20, engine=engine)
            assert len(split) == 1
            assert g.np.isclose(split[0].volume, 7.0)
            assert split[0].is_winding_consistent
            # with a low threshold both survive and the small shell is a
            # positive-volume solid, not a negative cavity
            split = mixed.split(solids=True, orient=True, min_faces=4, engine=engine)
            assert g.np.allclose(g.np.sort([i.volume for i in split]), [0.125, 7.0])

    def test_split_solids_append(self):
        # append=True must apply orientation flips at the right output offset
        for engine in self.engines:
            hollow = _box([2, 2, 2]) + _box([1, 1, 1], invert=True)
            outward_cavity = _box([2, 2, 2]) + _box([1, 1, 1])

            # already-correct input: no flips, single concatenated mesh
            appended = hollow.split(solids=True, orient=True, append=True, engine=engine)
            assert isinstance(appended, g.trimesh.Trimesh)
            assert appended.is_watertight
            assert appended.is_winding_consistent
            assert g.np.isclose(appended.volume, 7.0)

            # orientation actually flips faces here: a standalone inverted solid
            # forces a flip at offset 0 and the outward-wound cavity forces a
            # flip at a nonzero offset, exercising the offset arithmetic that
            # the already-correct `hollow` case above cannot
            inv = _box([1, 1, 1], invert=True).apply_translation([9, 0, 0])
            flipped = inv + outward_cavity
            appended = flipped.split(solids=True, orient=True, append=True, engine=engine)
            assert isinstance(appended, g.trimesh.Trimesh)
            assert appended.is_watertight
            assert appended.is_winding_consistent
            # inverted unit cube (1.0) + hollow 2-cube with 1-cube cavity (7.0)
            assert g.np.isclose(appended.volume, 8.0)

    def test_split_solids_soup(self):
        # a soup of disconnected triangles has no closed shells so the
        # solids path must fall through to the legacy component count
        soup = g.get_mesh("soup.stl")
        for engine in self.engines:
            legacy = soup.split(only_watertight=False, engine=engine)
            solids = soup.split(solids=True, only_watertight=False, engine=engine)
            assert len(solids) == len(legacy)

    def test_split_solids_contains(self):
        # direct unit test of the containment matrix helper: an axis-aligned
        # ray direction is the classic degenerate case for ray casting, so it
        # must produce the same correct matrix as the default direction
        graph = g.trimesh.graph
        np = g.np

        # three concentric axis-aligned boxes: shell ⊃ shell ⊃ shell
        mesh = _box([6, 6, 6]) + _box([4, 4, 4]) + _box([2, 2, 2])
        components = [
            np.asanyarray(c, dtype=np.int64)
            for c in graph.connected_components(
                mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
            )
        ]
        assert len(components) == 3

        # reuse the same setup `_split_solids` feeds `_contains_matrix`
        mins, maxs, points, candidate, labels = graph._shell_bounds(mesh, components)

        # order shells largest -> smallest so we know each nesting level
        volume = np.prod(maxs - mins, axis=1)
        outer, middle, inner = np.argsort(volume)[::-1]

        # the fixed default direction and a deliberately degenerate axis-aligned
        # direction must both produce the same, correct containment matrix
        for direction in (None, [0.0, 0.0, 1.0]):
            contains = graph._contains_matrix(
                mesh,
                labels=labels,
                points=points,
                candidate=candidate,
                direction=direction,
            )
            # every enclosing relationship is detected
            assert contains[outer, middle]
            assert contains[outer, inner]
            assert contains[middle, inner]
            # containment never runs backwards and no shell contains itself
            assert not contains[inner, outer]
            assert not contains[middle, outer]
            assert not contains[inner, middle]
            assert not contains.diagonal().any()

    def test_vertex_adjacency_graph(self):
        f = g.trimesh.graph.vertex_adjacency_graph

        # a mesh with a single watertight body
        sing = g.get_mesh("featuretype.STL")
        vert_adj_g = f(sing)
        assert len(sing.vertices) == len(vert_adj_g)

    def test_engine_time(self):
        for mesh in g.get_meshes():
            tic = [g.time.time()]
            for engine in self.engines:
                mesh.split(engine=engine, only_watertight=False)
                g.trimesh.graph.facets(mesh=mesh, engine=engine)
                tic.append(g.time.time())

            diff = g.np.abs(g.np.diff(tic))
            if diff.min() > 0.0:
                diff /= diff.min()

            g.log.info(
                "graph engine on %s (scale %f sec):\n%s",
                mesh.source.file_name,
                diff.min(),
                str(g.np.column_stack((self.engines, diff))),
            )

    def test_smoothed(self):
        # Make sure smoothing is keeping the same number
        # of faces.

        for name in ["ADIS16480.STL", "featuretype.STL"]:
            mesh = g.get_mesh(name)
            assert len(mesh.faces) == len(mesh.smooth_shaded.faces)

    def test_engines(self):
        edges = g.np.arange(10).reshape((-1, 2))
        for i in range(0, 20):
            check_engines(nodes=g.np.arange(i), edges=edges)
        edges = g.np.column_stack((g.np.arange(1, 11), g.np.arange(0, 10)))
        for i in range(0, 20):
            check_engines(nodes=g.np.arange(i), edges=edges)

    def test_watertight(self):
        m = g.get_mesh("shared.STL")  # NOQA
        # assert m.is_watertight
        # assert m.is_winding_consistent
        # assert m.is_volume

    def test_traversals(self):
        # Test traversals (BFS+DFS)

        # generate some simple test data
        simple_nodes = g.np.arange(20)
        simple_edges = g.np.column_stack((simple_nodes[:-1], simple_nodes[1:]))
        simple_edges = g.np.vstack(
            (simple_edges, [[19, 0], [10, 1000], [500, 501]])
        ).astype(g.np.int64)

        all_edges = g.data["edges"]
        all_edges.append(simple_edges)

        for edges in all_edges:
            edges = g.np.array(edges, dtype=g.np.int64)
            assert g.trimesh.util.is_shape(edges, (-1, 2))

            # collect the new nodes
            nodes = g.np.unique(edges)

            # the basic BFS/DFS traversal
            dfs_basic = g.trimesh.graph.traversals(edges, "dfs")
            bfs_basic = g.trimesh.graph.traversals(edges, "bfs")
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
                    [g.np.column_stack((i[:-1], i[1:])) for i in dfs]
                )

                # make a set from edges included in the traversal
                inc_set = {
                    i.tobytes()
                    for i in g.trimesh.grouping.hashable_rows(g.np.sort(inc, axis=1))
                }
                # make a set of the source edges we were supposed to include
                edge_set = {
                    i.tobytes()
                    for i in g.trimesh.grouping.hashable_rows(g.np.sort(edges, axis=1))
                }

                # we should have exactly the same edges
                # after the filled traversal as we started with
                assert len(inc) == len(edges)
                # every edge should occur exactly once
                assert len(inc_set) == len(inc)
                # unique edges should be the same
                assert inc_set == edge_set

                # check all return dtypes
                assert all(i.dtype == g.np.int64 for i in dfs)

    def test_traversal_no_fragmentation(self):
        """
        A DFS traversal of an open chain should not be split
        into fragments by fill_traversals. Previously, starting
        DFS from an interior node caused backtracking which
        produced non-edge consecutive pairs, fragmenting the path.
        """
        # simple open chain: 0-1-2-3-4-5-6-7
        chain = g.np.column_stack([g.np.arange(7), g.np.arange(1, 8)]).astype(g.np.int64)

        dfs = g.trimesh.graph.traversals(chain, mode="dfs")
        filled = g.trimesh.graph.fill_traversals(dfs, chain)

        # a single connected open chain must produce exactly 1 traversal
        assert len(filled) == 1
        # that traversal must contain all 8 nodes
        assert len(filled[0]) == 8

        # two disjoint chains should produce exactly 2 traversals
        chain2 = g.np.vstack(
            [
                chain,
                g.np.column_stack(
                    [
                        g.np.arange(100, 104),
                        g.np.arange(101, 105),
                    ]
                ).astype(g.np.int64),
            ]
        )
        dfs2 = g.trimesh.graph.traversals(chain2, mode="dfs")
        filled2 = g.trimesh.graph.fill_traversals(dfs2, chain2)
        assert len(filled2) == 2

    def test_adjacency(self):
        for add_degen in [False, True]:
            for name in ["featuretype.STL", "soup.stl"]:
                m = g.get_mesh(name)
                if add_degen:
                    # make the first face degenerate
                    m.faces[0][2] = m.faces[0][0]
                # degenerate faces should be filtered
                assert g.np.not_equal(*m.face_adjacency.T).all()

                # check the various paths of calling face adjacency
                a = g.trimesh.graph.face_adjacency(
                    m.faces.view(g.np.ndarray).copy(), return_edges=False
                )
                b, be = g.trimesh.graph.face_adjacency(
                    m.faces.view(g.np.ndarray).copy(), return_edges=True
                )
                c = g.trimesh.graph.face_adjacency(mesh=m, return_edges=False)
                c, ce = g.trimesh.graph.face_adjacency(mesh=m, return_edges=True)
                # make sure they all return the expected result
                assert g.np.allclose(a, b)
                assert g.np.allclose(a, c)
                assert len(be) == len(a)
                assert len(ce) == len(a)

                # package properties to loop through
                zips = zip(
                    m.face_adjacency, m.face_adjacency_edges, m.face_adjacency_unshared
                )
                for a, e, v in zips:
                    # get two adjacenct faces as a set
                    fa = set(m.faces[a[0]])
                    fb = set(m.faces[a[1]])

                    # face should be different
                    assert fa != fb
                    # shared edge should be in both faces

                    # removing 2 vertices should leave one
                    da = fa.difference(e)
                    db = fb.difference(e)
                    assert len(da) == 1
                    assert len(db) == 1

                    # unshared vertex should be correct
                    assert da.issubset(v)
                    assert db.issubset(v)
                    assert da != db
                    assert len(v) == 2


def check_engines(edges, nodes):
    """
    Make sure connected component graph engines are
    returning the exact same values
    """
    results = []
    engines = [None, "scipy", "networkx"]

    for engine in engines:
        c = g.trimesh.graph.connected_components(edges, nodes=nodes, engine=engine)
        if len(c) > 0:
            # check to see if every resulting component
            # was in the passed set of nodes
            diff = g.np.setdiff1d(g.np.hstack(c), nodes)
            assert len(diff) == 0
        # store the result as a set of tuples so we can compare
        results.append({tuple(sorted(i)) for i in c})

    # make sure different engines are returning the same thing
    try:
        assert all(i == results[0] for i in results[1:])
    except BaseException as E:
        g.log.debug(results)
        raise E


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
