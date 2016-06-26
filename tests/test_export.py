import generic as g

class ExportTest(g.unittest.TestCase):
    def test_export(self):
        file_types = list(g.trimesh.io.export._mesh_exporters.keys())
        for mesh in g.get_meshes(3):
            for file_type in file_types:
                e = mesh.export(file_type = file_type)
                self.assertTrue(len(e) > 0)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
