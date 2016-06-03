import generic as g

class VisualTest(g.unittest.TestCase):
    def test_visual(self):
        mesh = g.get_mesh('featuretype.STL')


        self.assertFalse(mesh.visual.defined)

        facets = mesh.facets()
        for facet in facets:
            mesh.visual.face_colors[facet] = g.trimesh.visual.random_color()

        self.assertTrue(mesh.visual.defined)
        self.assertFalse(mesh.visual.transparency)

        mesh.visual.face_colors[0] = [10,10,10,130]

        self.assertTrue(mesh.visual.transparency)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
