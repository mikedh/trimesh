import generic as g


class MeshTests(g.unittest.TestCase):

    def test_meshes(self):
        # make sure we can load everything we think we can
        # while getting a list of meshes to run tests on
        meshes = g.get_meshes(raise_error=True)
        g.log.info('Running tests on %d meshes', len(meshes))

        for mesh in meshes:
            g.log.info('Testing %s', mesh.metadata['file_name'])
            self.assertTrue(len(mesh.faces) > 0)
            self.assertTrue(len(mesh.vertices) > 0)

            self.assertTrue(len(mesh.edges) > 0)
            self.assertTrue(len(mesh.edges_unique) > 0)
            self.assertTrue(len(mesh.edges_sorted) > 0)
            self.assertTrue(len(mesh.edges_face) > 0)
            self.assertTrue(isinstance(mesh.euler_number, int))

            mesh.process()

            if not mesh.is_watertight:
                continue

            assert len(mesh.facets) == len(mesh.facets_area)
            assert len(mesh.facets) == len(mesh.facets_normal)
            assert len(mesh.facets) == len(mesh.facets_boundary)
            
            if len(mesh.facets) != 0:
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
            self.assertTrue(hasattr(t, 'query'))
            g.log.info('Creating triangles tree')
            r = mesh.triangles_tree()
            self.assertTrue(hasattr(r, 'intersection'))
            g.log.info('Triangles tree ok')

            # some memory issues only show up when you copy the mesh a bunch
            # specifically, if you cache c- objects then deepcopy the mesh this
            # generally segfaults randomly
            copy_count = 200
            g.log.info('Attempting to copy mesh %d times', copy_count)
            for i in range(copy_count):
                copied = mesh.copy()
            g.log.info('Multiple copies done')
            self.assertTrue(g.np.allclose(copied.identifier,
                                          mesh.identifier))

    def test_fill_holes(self):
        for mesh in g.get_meshes(5):
            if not mesh.is_watertight:
                continue
            mesh.faces = mesh.faces[1:-1]
            self.assertFalse(mesh.is_watertight)
            mesh.fill_holes()
            self.assertTrue(mesh.is_watertight)

    def test_fix_normals(self):
        for mesh in g.get_meshes(5):
            mesh.fix_normals()


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
