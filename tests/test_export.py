import generic as g

class ExportTest(g.unittest.TestCase):
    def test_export(self):
        file_types = list(g.trimesh.io.export._mesh_exporters.keys())
        for mesh in g.get_meshes(3):
            for file_type in file_types:
                export = mesh.export(file_type = file_type)
                self.assertTrue(len(export) > 0)
        
                # we don't have native loaders implemented for collada yet
                if file_type in ['dae', 'collada']:
                    continue

                if g.trimesh.util.is_string(export):
                    export = g.StringIO(export)
                loaded = g.trimesh.load(file_obj  = export,
                                        file_type = file_type)

                if loaded.faces.shape != mesh.faces.shape:
                    g.log.error('Export -> inport for %s on %s wrong shape!',
                                file_type, 
                                mesh.metadata['file_name'])
                self.assertTrue(loaded.faces.shape    == mesh.faces.shape)
                self.assertTrue(loaded.vertices.shape == mesh.vertices.shape)

                
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
