import generic as g


class ExportTest(g.unittest.TestCase):

    def test_export(self):
        file_types = list(g.trimesh.io.export._mesh_exporters.keys())
        for mesh in g.get_meshes(5):
            for file_type in file_types:
                export = mesh.export(file_type=file_type)

                if export is None:
                    raise ValueError('Exporting mesh %s to %s resulted in None!',
                                     mesh.metadata['file_name'],
                                     file_type)

                self.assertTrue(len(export) > 0)

                # we don't have native loaders implemented for collada yet
                # msgpack is kind of flaky but does *usually* work
                if file_type in ['dae', 'collada', 'msgpack']:
                    g.log.warning(
                        'Still no native loaders implemented for collada!')
                    continue

                g.log.info('Export/import testing on %s',
                           mesh.metadata['file_name'])
                loaded = g.trimesh.load(file_obj=g.io_wrap(export),
                                        file_type=file_type)

                if (not g.trimesh.util.is_shape(loaded._data['faces'],    (-1, 3)) or
                    not g.trimesh.util.is_shape(loaded._data['vertices'], (-1, 3)) or
                        loaded.faces.shape != mesh.faces.shape):
                    g.log.error('Export -> inport for %s on %s wrong shape!',
                                file_type,
                                mesh.metadata['file_name'])

                if loaded.vertices is None:
                    g.log.error('Export -> import for %s on %s gave None for vertices!',
                                file_type,
                                mesh.metadata['file_name'])
                self.assertTrue(loaded.faces.shape == mesh.faces.shape)
                self.assertTrue(loaded.vertices.shape == mesh.vertices.shape)
                g.log.info(
                    'Mesh vertices/faces consistent after export->import')


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
