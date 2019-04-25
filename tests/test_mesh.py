"""
Load all the meshes we can get our hands on and check things, stuff.
"""
try:
    from . import generic as g
except BaseException:
    import generic as g


class MeshTests(g.unittest.TestCase):

    def test_meshes(self):
        # make sure we can load everything we think we can
        formats = g.trimesh.available_formats()
        assert all(isinstance(i, str) for i in formats)
        assert all(len(i) > 0 for i in formats)
        assert all(i in formats
                   for i in ['stl', 'ply', 'off', 'obj'])

        for mesh in g.get_meshes(raise_error=True):
            # log file name for debugging
            file_name = mesh.metadata['file_name']
            g.log.info('Testing %s', file_name)

            start = {mesh.md5(), mesh.crc()}
            assert len(mesh.faces) > 0
            assert len(mesh.vertices) > 0

            assert len(mesh.edges) > 0
            assert len(mesh.edges_unique) > 0
            assert len(mesh.edges_sorted) > 0
            assert len(mesh.edges_face) > 0
            assert isinstance(mesh.euler_number, int)

            # check bounding primitives
            assert mesh.bounding_box.volume > 0.0
            assert mesh.bounding_primitive.volume > 0.0

            # none of these should have mutated anything
            assert start == {mesh.md5(), mesh.crc()}

            # run processing, again
            mesh.process()

            # still shouldn't have changed anything
            assert start == {mesh.md5(), mesh.crc()}

            if not (mesh.is_watertight and
                    mesh.is_winding_consistent):
                continue

            assert len(mesh.facets) == len(mesh.facets_area)
            assert len(mesh.facets) == len(mesh.facets_normal)
            assert len(mesh.facets) == len(mesh.facets_boundary)

            if len(mesh.facets) != 0:
                faces = mesh.facets[mesh.facets_area.argmax()]
                outline = mesh.outline(faces)
                # check to make sure we can generate closed paths
                # on a Path3D object
                test = outline.paths  # NOQA

            smoothed = mesh.smoothed()  # NOQA

            assert mesh.volume > 0.0

            section = mesh.section(plane_normal=[0, 0, 1],  # NOQA
                                   plane_origin=mesh.centroid)

            sample = mesh.sample(1000)
            even_sample = g.trimesh.sample.sample_surface_even(mesh, 100)  # NOQA
            assert sample.shape == (1000, 3)
            g.log.info('finished testing meshes')

            # make sure vertex kdtree and triangles rtree exist

            t = mesh.kdtree
            assert hasattr(t, 'query')
            g.log.info('Creating triangles tree')
            r = mesh.triangles_tree
            assert hasattr(r, 'intersection')
            g.log.info('Triangles tree ok')

            # face angles should have same
            assert mesh.face_angles.shape == mesh.faces.shape
            assert len(mesh.vertices) == len(mesh.vertex_defects)
            assert len(mesh.principal_inertia_components) == 3

            # we should have built up a bunch of stuff into
            # our cache, so make sure all numpy arrays cached are
            # finite
            for name, cached in mesh._cache.cache.items():
                # only check numpy arrays
                if not isinstance(cached, g.np.ndarray):
                    continue

                # only check int, float, and bool
                if cached.dtype.kind not in 'ibf':
                    continue

                # there should never be NaN values
                if g.np.isnan(cached).any():
                    raise ValueError('NaN values in %s/%s',
                                     file_name, name)

                # fields allowed to have infinite values
                if name in ['face_adjacency_radius']:
                    continue

                # make sure everything is finite
                if not g.np.isfinite(cached).all():
                    raise ValueError('inf values in %s/%s',
                                     file_name, name)

            # some memory issues only show up when you copy the mesh a bunch
            # specifically, if you cache c- objects then deepcopy the mesh this
            # generally segfaults randomly
            copy_count = 200
            g.log.info('Attempting to copy mesh %d times', copy_count)
            for i in range(copy_count):
                copied = mesh.copy()
                assert copied.is_empty == mesh.is_empty
                # t = copied.triangles_tree
                c = copied.kdtree  # NOQA
                copied.apply_transform(
                    g.trimesh.transformations.rotation_matrix(
                        g.np.degrees(i),
                        [0, 1, 1]))
            g.log.info('Multiple copies done')

            if not g.np.allclose(copied.identifier,
                                 mesh.identifier):
                raise ValueError('copied identifier changed!')

            # ...still shouldn't have changed anything
            assert start == {mesh.md5(), mesh.crc()}


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
