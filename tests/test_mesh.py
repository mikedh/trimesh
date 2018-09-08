try:
    from . import generic as g
except BaseException:
    import generic as g


class MeshTests(g.unittest.TestCase):

    def test_meshes(self):
        # make sure we can load everything we think we can
        # while getting a list of meshes to run tests on
        meshes = g.get_meshes(raise_error=True)
        g.log.info('Running tests on %d meshes', len(meshes))

        formats = g.trimesh.available_formats()
        assert all(isinstance(i, str) for i in formats)
        assert all(len(i) > 0 for i in formats)

        for mesh in meshes:
            g.log.info('Testing %s', mesh.metadata['file_name'])
            assert len(mesh.faces) > 0
            assert len(mesh.vertices) > 0

            assert len(mesh.edges) > 0
            assert len(mesh.edges_unique) > 0
            assert len(mesh.edges_sorted) > 0
            assert len(mesh.edges_face) > 0
            assert isinstance(mesh.euler_number, int)

            mesh.process()

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
                test = outline.paths

            smoothed = mesh.smoothed()

            assert mesh.volume > 0.0

            section = mesh.section(plane_normal=[0, 0, 1],
                                   plane_origin=mesh.centroid)

            sample = mesh.sample(1000)
            even_sample = g.trimesh.sample.sample_surface_even(mesh, 100)
            assert sample.shape == (1000, 3)
            g.log.info('finished testing meshes')

            # make sure vertex kdtree and triangles rtree exist

            t = mesh.kdtree
            assert hasattr(t, 'query')
            g.log.info('Creating triangles tree')
            r = mesh.triangles_tree
            assert hasattr(r, 'intersection')
            g.log.info('Triangles tree ok')

            # some memory issues only show up when you copy the mesh a bunch
            # specifically, if you cache c- objects then deepcopy the mesh this
            # generally segfaults randomly
            copy_count = 200
            g.log.info('Attempting to copy mesh %d times', copy_count)
            for i in range(copy_count):
                copied = mesh.copy()
                assert copied.is_empty == mesh.is_empty
                #t = copied.triangles_tree
                c = copied.kdtree
                copied.apply_transform(
                    g.trimesh.transformations.rotation_matrix(
                        g.np.degrees(i),
                        [0, 1, 1]))
            g.log.info('Multiple copies done')

            if not g.np.allclose(copied.identifier,
                                 mesh.identifier):
                raise ValueError('copied identifier changed!')

    def test_vertex_neighbors(self):
        m = g.trimesh.primitives.Box()
        neighbors = m.vertex_neighbors
        assert len(neighbors) == len(m.vertices)
        elist = m.edges_unique.tolist()

        for v_i, neighs in enumerate(neighbors):
            for n in neighs:
                assert ([v_i, n] in elist or [n, v_i] in elist)


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
