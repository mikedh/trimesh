try:
    from . import generic as g
except BaseException:
    import generic as g


class ExportTest(g.unittest.TestCase):

    def test_export(self):
        export_types = list(g.trimesh.io.export._mesh_exporters.keys())
        for mesh in g.get_meshes(8):
            for file_type in export_types:
                export = mesh.export(file_type=file_type)
                if export is None:
                    raise ValueError('Exporting mesh %s to %s resulted in None!',
                                     mesh.metadata['file_name'],
                                     file_type)

                assert len(export) > 0

                if file_type in [
                        'dae',     # collada, no native importers
                        'collada',  # collada, no native importers
                        'msgpack',  # kind of flaky, but usually works
                        'drc']:    # DRC is not a lossless format
                    g.log.warning(
                        'no native loaders implemented for collada!')
                    continue

                g.log.info('Export/import testing on %s',
                           mesh.metadata['file_name'])

                # if export is string or bytes wrap as pseudo file object
                if isinstance(export, str) or isinstance(export, bytes):
                    file_obj = g.io_wrap(export)
                else:
                    file_obj = export

                loaded = g.trimesh.load(file_obj=file_obj,
                                        file_type=file_type)

                if (not g.trimesh.util.is_shape(loaded._data['faces'], (-1, 3)) or
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
                    raise ValueError('export cycle {} on {} gave vertices {}->{}!'.format(
                        file_type,
                        mesh.metadata['file_name'],
                        str(mesh.faces.shape),
                        str(loaded.faces.shape)))
                assert loaded.vertices.shape == mesh.vertices.shape

                # try exporting/importing certain file types by name
                if file_type in ['obj', 'stl', 'ply', 'off']:
                    temp = g.tempfile.NamedTemporaryFile(suffix='.' + file_type,
                                                         delete=False)
                    # windows throws permissions errors if you keep it open
                    temp.close()

                    mesh.export(temp.name)
                    load = g.trimesh.load(temp.name)
                    # manual cleanup
                    g.os.remove(temp.name)

                    assert mesh.faces.shape == load.faces.shape
                    assert mesh.vertices.shape == load.vertices.shape

            # if we're not on linux don't run meshlab tests
            if not g.is_linux:
                continue
            # formats exportable by trimesh and importable by meshlab
            # make sure things we export can be loaded by meshlab
            both = set(g.meshlab_formats).intersection(
                set(export_types))

            # additional options to pass to exporters to try to ferret
            # out combinations which lead to invalid output
            kwargs = {'ply': [{'vertex_normal': True,
                               'encoding': 'ascii'},
                              {'vertex_normal': True,
                               'encoding': 'binary'},
                              {'vertex_normal': False,
                               'encoding': 'ascii'},
                              {'vertex_normal': False,
                               'encoding': 'binary'}],
                      'stl': [{'file_type': 'stl'},
                              {'file_type': 'stl_ascii'}]}

            # make sure input mesh has garbage removed
            mesh._validate = True
            # since we're going to be looking for exact export
            # counts remove anything small/degenerate again
            mesh.process()

            # run through file types supported by both meshlab and trimesh
            for file_type in both:
                # pull different exporter options for the format
                if file_type in kwargs:
                    options = kwargs[file_type]
                else:
                    options = [{}]

                # try each combination of options
                for option in options:
                    temp = g.tempfile.NamedTemporaryFile(
                        suffix='.' + file_type,
                        delete=False)
                    temp_off = g.tempfile.NamedTemporaryFile(
                        suffix='.off',
                        delete=False)
                    # windows throws permissions errors if you keep it open
                    temp.close()
                    temp_off.close()
                    # write over the tempfile
                    option['file_obj'] = temp.name
                    mesh.export(**option)

                    # will raise CalledProcessError if meshlab
                    # can't successfully import the file
                    try:
                        # have meshlab take the export and convert it into
                        # an OFF file, which is basically the simplest format
                        # that uses by- reference vertices
                        # meshlabserver requires X so wrap it with XVFB
                        cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver '
                        cmd += '-i {} -o {}'.format(temp.name, temp_off.name)
                        g.subprocess.check_call(cmd, shell=True)
                    except g.subprocess.CalledProcessError as E:
                        # log the options that produced the failure
                        g.log.error('failed to export {}'.format(
                            option))
                        # raise the error again
                        raise E

                    # load meshlabs export back into trimesh
                    r = g.trimesh.load(temp_off.name)

                    # we should have the same number of vertices and faces
                    assert len(r.vertices) == len(mesh.vertices)
                    assert len(r.faces) == len(mesh.faces)

                    # manual cleanup
                    g.os.remove(temp.name)
                    g.os.remove(temp_off.name)

    def test_obj(self):
        m = g.get_mesh('textured_tetrahedron.obj', process=False)
        export = m.export(file_type='obj')
        reconstructed = g.trimesh.load(g.trimesh.util.wrap_as_stream(export),
                                       file_type='obj', process=False)
        # test that we get at least the same number of normals and texcoords out;
        # the loader may reorder vertices, so we shouldn't check direct
        # equality
        assert m.vertex_normals.shape == reconstructed.vertex_normals.shape
        assert m.metadata['vertex_texture'].shape == reconstructed.metadata[
            'vertex_texture'].shape

    def test_obj_order(self):
        """
        Make sure simple round trips through Wavefront don't
        reorder vertices.
        """
        # get a writeable temp file location
        temp = g.tempfile.NamedTemporaryFile(
            suffix='.obj',
            delete=False)
        temp.close()

        # simple solid
        x = g.trimesh.creation.icosahedron()
        x.export(temp.name)
        y = g.trimesh.load_mesh(temp.name, process=False)

        # vertices should be same shape and order
        assert g.np.allclose(x.vertices, y.vertices)
        # faces should be same
        assert g.np.allclose(x.faces, y.faces)

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

    def test_scene(self):
        # get a multi- mesh scene with a transform tree
        source = g.get_mesh('cycloidal.3DXML')
        # add a transform to zero scene before exporting
        source.rezero()
        # export the file as a binary GLTF file, GLB
        export = source.export(file_type='glb')

        # re- load the file as a trimesh.Scene object again
        loaded = g.trimesh.load(
            file_obj=g.trimesh.util.wrap_as_stream(export),
            file_type='glb')

        # the scene should be identical after export-> import cycle
        assert g.np.allclose(loaded.extents / source.extents,
                             1.0)

    def test_gltf_path(self):
        """
        Check to make sure GLTF exports of Path2D and Path3D
        objects don't immediatly crash.
        """
        path2D = g.get_mesh('2D/wrench.dxf')
        path3D = path2D.to_3D()

        a = g.trimesh.Scene(path2D).export(file_type='glb')
        b = g.trimesh.Scene(path3D).export(file_type='glb')

        assert len(a) > 0
        assert len(b) > 0

    def test_parse_file_args(self):
        """
        Test the magical trimesh.io.load.parse_file_args
        """
        # it's wordy
        f = g.trimesh.io.load.parse_file_args

        # a path that doesn't exist
        nonexists = '/banana{}'.format(g.np.random.random())
        assert not g.os.path.exists(nonexists)

        # loadable OBJ model
        exists = g.os.path.join(g.dir_models, 'tube.obj')
        assert g.os.path.exists(exists)

        # should be able to extract type from passed filename
        args = f(file_obj=exists, file_type=None)
        assert len(args) == 4
        assert args[1] == 'obj'

        # should be able to extract correct type from longer name
        args = f(file_obj=exists, file_type='YOYOMA.oBj')
        assert len(args) == 4
        assert args[1] == 'obj'

        # with a nonexistant file and no extension it should raise
        try:
            args = f(file_obj=nonexists, file_type=None)
        except ValueError as E:
            assert 'not a file' in str(E)
        else:
            raise ValueError('should have raised exception!')

        # nonexistant file with extension passed should return
        # file name anyway, maybe something else can handle i
        args = f(file_obj=nonexists, file_type='.ObJ')
        assert len(args) == 4
        # should have cleaned up case
        assert args[1] == 'obj'

        # make sure overriding type works for string filenames
        args = f(file_obj=exists, file_type='STL')
        assert len(args) == 4
        # should have used manually passed type over .obj
        assert args[1] == 'stl'


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
