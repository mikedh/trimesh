try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.scene.transforms import EnforcedForest


def random_chr():
    return chr(ord('a') + int(round(g.np.random.random() * 25)))


class GraphTests(g.unittest.TestCase):

    def test_forest(self):
        graph = EnforcedForest()
        for i in range(5000):
            graph.add_edge(random_chr(), random_chr())

    def test_cache(self):
        for i in range(10):
            scene = g.trimesh.Scene()
            scene.add_geometry(g.trimesh.creation.box())

            mod = [scene.graph.modified()]
            scene.set_camera()
            mod.append(scene.graph.modified())
            assert mod[-1] != mod[-2]

            assert not g.np.allclose(
                scene.camera_transform,
                g.np.eye(4))
            scene.camera_transform = g.np.eye(4)
            mod.append(scene.graph.modified())
            assert mod[-1] != mod[-2]

            assert g.np.allclose(
                scene.camera_transform,
                g.np.eye(4))
            assert mod[-1] != mod[-2]

    def test_successors(self):
        s = g.get_mesh('CesiumMilkTruck.glb')
        assert len(s.graph.nodes_geometry) == 5

        # world should be root frame
        assert (s.graph.transforms.successors(
            s.graph.base_frame) == s.graph.nodes)

        for n in s.graph.nodes:
            # successors should always return subset of nodes
            succ = s.graph.transforms.successors(n)
            assert succ.issubset(
                s.graph.nodes)
            # we self-include node in successors
            assert n in succ

        # test getting a subscene from successors
        ss = s.subscene('3')
        assert len(ss.geometry) == 1
        assert len(ss.graph.nodes_geometry) == 1

        assert isinstance(s.graph.to_networkx(),
                          g.nx.DiGraph)

    def test_nodes(self):
        # get a scene graph
        graph = g.get_mesh('models/cycloidal.3DXML').graph
        # get any non-root node
        node = list(graph.nodes.difference(['world']))[0]
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
        fix = g.np.random.random((4, 4))
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

        ss = s.subscene('3')

        assert ss.graph.base_frame == '3'
        assert len(ss.graph.nodes) == 2
        assert ss.graph.nodes == {'3', '4'}
        assert len(ss.graph.transforms.node_data) == 2
        assert len(ss.graph.transforms.edge_data) == 1
        assert list(ss.graph.transforms.edge_data.keys()) == [('3', '4')]


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
