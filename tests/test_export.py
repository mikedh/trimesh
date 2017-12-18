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

                if file_type in ['dae',     # collada, no native importers
                                 'collada',  # collada, no native importers
                                 'msgpack',  # kind of flaky, but usually works
                                 'drc']:    # DRC is not a lossless format
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
                    g.log.error('Export -> import for %s on %s wrong shape!',
                                file_type,
                                mesh.metadata['file_name'])

                if loaded.vertices is None:
                    g.log.error('Export -> import for %s on %s gave None for vertices!',
                                file_type,
                                mesh.metadata['file_name'])

                if loaded.faces.shape != mesh.faces.shape:
                    raise ValueError('Export -> import for {} on {} gave vertices {}->{}!'.format(
                        file_type,
                        mesh.metadata['file_name'],
                        str(mesh.faces.shape),
                        str(loaded.faces.shape)))
                self.assertTrue(loaded.vertices.shape == mesh.vertices.shape)

    def test_obj(self):
        m = g.get_mesh('textured_tetrahedron.obj', process=False)
        export = m.export(file_type='obj')
        reconstructed = g.trimesh.load(g.trimesh.util.wrap_as_stream(export),
                                       file_type='obj', process=False)
        # test that we get at least the same number of normals and texcoords out;
        # the loader may reorder vertices, so we shouldn't check direct equality
        assert m.vertex_normals.shape == reconstructed.vertex_normals.shape
        assert m.metadata['vertex_texture'].shape == reconstructed.metadata['vertex_texture'].shape

    def test_ply(self):
        m = g.get_mesh('machinist.XAML')

        assert m.visual.kind == 'face'
        assert m.visual.face_colors.ptp(axis=0).max() > 0

        export = m.export(file_type='ply')
        reconstructed = g.trimesh.load(g.trimesh.util.wrap_as_stream(export),
                                       file_type='ply')

        assert reconstructed.visual.kind == 'face'

        assert g.np.allclose(reconstructed.visual.face_colors,
                             m.visual.face_colors)

        m = g.get_mesh('reference.ply')

        assert m.visual.kind == 'vertex'
        assert m.visual.vertex_colors.ptp(axis=0).max() > 0

        export = m.export(file_type='ply')
        reconstructed = g.trimesh.load(g.trimesh.util.wrap_as_stream(export),
                                       file_type='ply')

        assert reconstructed.visual.kind == 'vertex'

        assert g.np.allclose(reconstructed.visual.vertex_colors,
                             m.visual.vertex_colors)


    def test_dict(self):
        mesh = g.get_mesh('machinist.XAML')
        assert mesh.visual.kind == 'face'
        mesh.visual.vertex_colors = mesh.visual.vertex_colors
        assert mesh.visual.kind == 'vertex'

        as_dict = mesh.to_dict()
        back = g.trimesh.Trimesh(**as_dict)

        

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
