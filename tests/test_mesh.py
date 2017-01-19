import generic as g

from collections import deque 

class MeshTests(g.unittest.TestCase):
    def test_meshes(self):
        self.meshes = g.get_meshes()

        g.log.info('Running tests on %d meshes', len(self.meshes))
        for mesh in self.meshes:
            g.log.info('Testing %s', mesh.metadata['file_name'])
            self.assertTrue(len(mesh.faces) > 0)
            self.assertTrue(len(mesh.vertices) > 0)
            
            self.assertTrue(len(mesh.edges) > 0)
            self.assertTrue(len(mesh.edges_unique) > 0)
            self.assertTrue(len(mesh.edges_sorted) > 0)
            self.assertTrue(len(mesh.edges_face) > 0)
            self.assertTrue(isinstance(mesh.euler_number, int))

            mesh.process()


            facets, area = mesh.facets(return_area=True)
            self.assertTrue(len(facets) == len(area))
            if len(facets) == 0:
                continue
                
            faces = facets[g.np.argmax(area)]
            outline = mesh.outline(faces)
            smoothed = mesh.smoothed()
                
            self.assertTrue(mesh.volume > 0.0)
                
            section   = mesh.section(plane_normal=[0,0,1], plane_origin=mesh.centroid)
            hull      = mesh.convex_hull

            volume_ok = hull.volume > 0.0
            if not volume_ok:
                g.log.error('zero hull volume for %s', mesh.metadata['file_name'])
            self.assertTrue(volume_ok)

            sample = mesh.sample(1000)
            even_sample = g.trimesh.sample.sample_surface_even(mesh, 100)
            self.assertTrue(sample.shape == (1000,3))
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
            # generally segfaults somewhat randomly
            copy_count = 200
            g.log.info('Attempting to copy mesh %d times', copy_count)
            for i in range(copy_count):
                copied = mesh.copy()
            g.log.info('Multiple copies done')
            self.assertTrue(g.np.allclose(copied.identifier,
                                          mesh.identifier))
            
    def test_fill_holes(self):
        for mesh in g.get_meshes(5):
            if not mesh.is_watertight: continue
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
