import generic as g

from trimesh.scene.transforms import EnforcedForest

def random_chr():
    return chr(ord('a') + int(round(g.np.random.random() * 25)))


class SceneTests(g.unittest.TestCase):
    def test_scene(self):
        for mesh in g.get_mesh('cycloidal.ply',
                               'kinematic.tar.gz',
                               'sphere.ply'):
            scene_split = g.trimesh.scene.split_scene(mesh)
            

            scene_base = g.trimesh.Scene(mesh)
            
            for s in [scene_split, scene_base]:
                self.assertTrue(len(s.geometry) > 0)
                
                flattened = s.transforms.to_flattened()
                g.json.dumps(flattened)
                edgelist = s.transforms.to_edgelist()
                g.json.dumps(edgelist)
            
class GraphTests(g.unittest.TestCase):

    def test_forest(self):
        g = EnforcedForest(assert_forest=True)
        for i in range(5000):
            g.add_edge(random_chr(), random_chr())


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
