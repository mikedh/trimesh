import generic as g




def typical_application():
    # make sure we can load everything we think we can
    # while getting a list of meshes to run tests on
    meshes = g.get_meshes(raise_error=True)
    g.log.info('Running tests on %d meshes', len(meshes))

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

        if not mesh.is_watertight:
            continue

        assert len(mesh.facets) == len(mesh.facets_area)
        if len(mesh.facets) == 0:
            continue

        faces = mesh.facets[mesh.facets_area.argmax()]
        outline = mesh.outline(faces)
        smoothed = mesh.smoothed()

        assert mesh.volume > 0.0

        section = mesh.section(plane_normal=[0, 0, 1],
                               plane_origin=mesh.centroid)

        sample = mesh.sample(1000)
        even_sample = g.trimesh.sample.sample_surface_even(mesh, 100)
        assert sample.shape == (1000, 3)
        g.log.info('finished testing meshes')

        # make sure vertex kdtree and triangles rtree exist

        t = mesh.kdtree()
        assert hasattr(t, 'query')
        g.log.info('Creating triangles tree')
        r = mesh.triangles_tree()
        assert hasattr(r, 'intersection')
        g.log.info('Triangles tree ok')

        # some memory issues only show up when you copy the mesh a bunch
        # specifically, if you cache c- objects then deepcopy the mesh this
        # generally segfaults randomly
        copy_count = 20
        g.log.info('Attempting to copy mesh %d times', copy_count)
        for i in range(copy_count):
            copied = mesh.copy()
        g.log.info('Multiple copies done')
        assert g.np.allclose(copied.identifier,
                             mesh.identifier)
        assert isinstance(mesh.identifier_md5, str)
        

if __name__ == '__main__':

    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

    typical_application()
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
