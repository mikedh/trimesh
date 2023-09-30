try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.transforms import EnforcedForest


def random_chr():
    return chr(ord("a") + int(round(g.random() * 25)))


class GraphTests(g.unittest.TestCase):
    def test_forest(self):
        graph = EnforcedForest()
        for _i in range(5000):
            graph.add_edge(random_chr(), random_chr())

    def test_cache(self):
        for _i in range(10):
            scene = g.trimesh.Scene()
            scene.add_geometry(g.trimesh.creation.box())

            mod = [scene.graph.__hash__()]
            scene.set_camera()
            mod.append(scene.graph.__hash__())
            assert mod[-1] != mod[-2]

            assert not g.np.allclose(scene.camera_transform, g.np.eye(4))
            scene.camera_transform = g.np.eye(4)
            mod.append(scene.graph.__hash__())
            assert mod[-1] != mod[-2]

            assert g.np.allclose(scene.camera_transform, g.np.eye(4))
            assert mod[-1] != mod[-2]

    def test_successors(self):
        s = g.get_mesh("CesiumMilkTruck.glb")
        assert len(s.graph.nodes_geometry) == 5

        # world should be root frame
        assert s.graph.transforms.successors(s.graph.base_frame) == set(s.graph.nodes)

        for n in s.graph.nodes:
            # successors should always return subset of nodes
            succ = s.graph.transforms.successors(n)
            assert succ.issubset(s.graph.nodes)
            # we self-include node in successors
            assert n in succ

        # test getting a subscene from successors
        ss = s.subscene("3")
        assert len(ss.geometry) == 1
        assert len(ss.graph.nodes_geometry) == 1

        assert isinstance(s.graph.to_networkx(), g.nx.DiGraph)

    def test_nodes(self):
        # get a scene graph
        graph = g.get_mesh("cycloidal.3DXML").graph
        # get any non-root node
        node = next(iter(set(graph.nodes).difference([graph.base_frame])))
        # remove that node
        graph.transforms.remove_node(node)
        # should have dumped the cache and removed the node
        assert node not in graph.nodes

    def test_kwargs(self):
        # test the function that converts various
        # arguments into a homogeneous transformation
        f = g.trimesh.scene.transforms.kwargs_to_matrix
        # no arguments should be an identity matrix
        assert g.np.allclose(f(), g.np.eye(4))

        # a passed matrix should return immediately
        fix = g.random((4, 4))
        assert g.np.allclose(f(matrix=fix), fix)

        quat = g.trimesh.unitize([1, 2, 3, 1])
        trans = [1.0, 2.0, 3.0]
        rot = g.trimesh.transformations.quaternion_matrix(quat)
        # should be the same as passed to transformations
        assert g.np.allclose(rot, f(quaternion=quat))

        # try passing both quaternion and translation
        combine = f(quaternion=quat, translation=trans)
        # should be the same as passed and computed
        assert g.np.allclose(combine[:3, :3], rot[:3, :3])
        assert g.np.allclose(combine[:3, 3], trans)

    def test_remove_node(self):
        s = g.get_mesh("CesiumMilkTruck.glb")

        assert len(s.graph.nodes_geometry) == 5
        assert len(s.graph.nodes) == 9
        assert len(s.graph.transforms.node_data) == 9
        assert len(s.graph.transforms.edge_data) == 8
        assert len(s.graph.transforms.parents) == 8

        assert s.graph.transforms.remove_node("1")

        assert len(s.graph.nodes_geometry) == 5
        assert len(s.graph.nodes) == 8
        assert len(s.graph.transforms.node_data) == 8
        assert len(s.graph.transforms.edge_data) == 6
        assert len(s.graph.transforms.parents) == 6

    def test_subscene(self):
        s = g.get_mesh("CesiumMilkTruck.glb")

        assert len(s.graph.nodes) == 9
        assert len(s.graph.transforms.node_data) == 9
        assert len(s.graph.transforms.edge_data) == 8

        ss = s.subscene("3")

        assert ss.graph.base_frame == "3"
        assert set(ss.graph.nodes) == {"3", "4"}
        assert len(ss.graph.transforms.node_data) == 2
        assert len(ss.graph.transforms.edge_data) == 1
        assert list(ss.graph.transforms.edge_data.keys()) == [("3", "4")]

    def test_scene_transform(self):
        # get a scene graph
        scene = g.get_mesh("cycloidal.3DXML")

        # copy the original bounds of the scene's convex hull
        b = scene.convex_hull.bounds.tolist()
        # dump it into a single mesh
        m = scene.dump(concatenate=True)

        # mesh bounds should match exactly
        assert g.np.allclose(m.bounds, b)
        assert g.np.allclose(scene.convex_hull.bounds, b)

        # get a random rotation matrix
        T = g.trimesh.transformations.random_rotation_matrix()

        # apply it to both the mesh and the scene
        m.apply_transform(T)
        scene.apply_transform(T)

        # the mesh and scene should have the same bounds
        assert g.np.allclose(m.convex_hull.bounds, scene.convex_hull.bounds)
        # should have moved from original position
        assert not g.np.allclose(m.convex_hull.bounds, b)

    def test_reverse(self):
        tf = g.trimesh.transformations

        s = g.trimesh.scene.Scene()
        s.add_geometry(
            g.trimesh.creation.box(),
            parent_node_name="world",
            node_name="foo",
            transform=tf.translation_matrix([0, 0, 1]),
        )

        s.add_geometry(
            g.trimesh.creation.box(),
            parent_node_name="foo",
            node_name="foo2",
            transform=tf.translation_matrix([0, 0, 1]),
        )

        assert len(s.graph.transforms.edge_data) == 2
        a = s.graph.get(frame_from="world", frame_to="foo2")

        assert len(s.graph.transforms.edge_data) == 2

        # try going backward
        i = s.graph.get(frame_from="foo2", frame_to="world")
        # matrix should be inverted if you're going the other way
        assert g.np.allclose(a[0], g.np.linalg.inv(i[0]))

        # try getting foo2 with shorthand
        b = s.graph.get(frame_to="foo2")
        c = s.graph["foo2"]
        # matrix should be inverted if you're going the other way
        assert g.np.allclose(a[0], c[0])
        assert g.np.allclose(b[0], c[0])

        # get should not have edited edge data
        assert len(s.graph.transforms.edge_data) == 2

    def test_shortest_path(self):
        # compare the EnforcedForest shortest path algo
        # to the more general networkx.shortest_path algo
        if g.sys.version_info < (3, 7):
            # old networkx is a lot different
            return

        tf = g.trimesh.transformations
        # start with creating a random tree
        edgelist = {}
        tree = g.nx.random_tree(n=1000, seed=0, create_using=g.nx.DiGraph)
        edges = list(tree.edges)

        r_choices = g.random((len(edges), 2))
        r_matrices = g.random_transforms(len(edges))
        for e, r_choice, r_mat in zip(edges, r_choices, r_matrices):
            data = {}
            if r_choice[0] > 0.5:
                # if a matrix is omitted but an edge exists it is
                # the same as passing an identity matrix
                data["matrix"] = r_mat
            if r_choice[1] > 0.4:
                # a geometry is not required for a node
                data["geometry"] = str(int(r_choice[1] * 1e8))
            edgelist[e] = data

        # now apply the random data to an EnforcedForest
        forest = g.trimesh.scene.transforms.EnforcedForest()
        for k, v in edgelist.items():
            forest.add_edge(*k, **v)

        # generate a lot of random queries
        queries = g.np.random.choice(list(forest.nodes), 10000).reshape((-1, 2))
        # filter out any self-queries as networkx doesn't handle them
        queries = queries[queries.ptp(axis=1) > 0]

        # now run our shortest path algorithm in a profiler
        with g.Profiler() as P:
            ours = [forest.shortest_path(*q) for q in queries]
        # print this way to avoid a python2 syntax error
        g.log.debug(P.output_text())

        # check truth from networkx with an undirected graph
        undir = tree.to_undirected()
        with g.Profiler() as P:
            truth = [g.nx.shortest_path(undir, *q) for q in queries]
        g.log.debug(P.output_text())

        # now compare our shortest path with networkx
        for a, b, q in zip(truth, ours, queries):
            if tuple(a) != tuple(b):
                # raise the query that killed us
                raise ValueError(q)

        # now try creating this as a full scenegraph
        sg = g.trimesh.scene.transforms.SceneGraph()
        [
            sg.update(frame_from=k[0], frame_to=k[1], **kwargs)
            for k, kwargs in edgelist.items()
        ]

        with g.Profiler() as P:
            matgeom = [sg.get(frame_from=q[0], frame_to=q[1]) for q in queries]
        g.log.debug(P.output_text())

        # all of the matrices should be rigid transforms
        assert all(tf.is_rigid(mat) for mat, _ in matgeom)

    def test_scaling_order(self):
        s = g.trimesh.creation.box().scene()
        scaling = 1.0 / 3.0
        c = s.scaled(scaling)
        factor = c.geometry["geometry_0"].vertices / s.geometry["geometry_0"].vertices
        assert g.np.allclose(factor, scaling)
        # should be returning itself
        r = s.apply_translation([10.5, 10.5, 10.5])
        assert g.np.allclose(r.bounds, [[10, 10, 10], [11, 11, 11]])
        assert g.np.allclose(s.bounds, [[10, 10, 10], [11, 11, 11]])

    def test_translation_cache(self):
        # scene with non-geometry nodes
        c = g.get_mesh("cycloidal.3DXML")
        s = c.scaled(1.0 / c.extents)
        # get the pre-translation bounds
        ori = s.bounds.copy()
        # apply a translation
        s.apply_translation([10, 10, 10])
        assert g.np.allclose(s.bounds, ori + 10)

    def test_translation_origin(self):
        # check to see if we can translate to the origin
        c = g.get_mesh("cycloidal.3DXML")
        c.apply_transform(g.trimesh.transformations.random_rotation_matrix())
        s = c.scaled(1.0 / c.extents)
        # shouldn't be at the origin
        assert not g.np.allclose(s.bounds[0], 0.0)
        # should move to the origin
        s.apply_translation(-s.bounds[0])
        assert g.np.allclose(s.bounds[0], 0)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
