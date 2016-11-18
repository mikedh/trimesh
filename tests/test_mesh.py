import generic as g

from collections import deque 

class MeshTests(g.unittest.TestCase):
    def test_meshes(self):
        self.meshes = g.get_meshes()

        has_gt = g.trimesh.graph._has_gt
        g.trimesh.graph._has_gt = False

        if not has_gt:
            g.log.warning('No graph-tool to test!')

        g.log.info('Running tests on %d meshes', len(self.meshes))
        for mesh in self.meshes:
            g.log.info('Testing %s', mesh.metadata['file_name'])
            self.assertTrue(len(mesh.faces) > 0)
            self.assertTrue(len(mesh.vertices) > 0)
            
            self.assertTrue(len(mesh.edges) > 0)
            self.assertTrue(len(mesh.edges_unique) > 0)
            self.assertTrue(len(mesh.edges_sorted) > 0)
            self.assertTrue(len(mesh.edges_face) > 0)
            self.assertFalse(mesh.euler_number is None)

            mesh.process()

            tic = [g.time.time()]

            if has_gt:
                g.trimesh.graph._has_gt = True 
                split     = g.trimesh.graph.split(mesh)
                tic.append(g.time.time())
                facets    = g.trimesh.graph.facets(mesh)
                tic.append(g.time.time())
                g.trimesh.graph._has_gt = False

            split     = g.trimesh.graph.split(mesh) 
            tic.append(g.time.time())
            facets    = g.trimesh.graph.facets(mesh)
            tic.append(g.time.time())

            facets, area = mesh.facets(return_area=True)
            self.assertTrue(len(facets) == len(area))
            if len(facets) == 0:
                continue
            faces = facets[g.np.argmax(area)]
            outline = mesh.outline(faces)
            smoothed = mesh.smoothed()

            if has_gt:
                times = g.np.diff(tic)
                g.log.info('Graph-tool sped up split by %f and facets by %f', 
                         (times[2] / times[0]), (times[3] / times[1]))

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
            r = mesh.triangles_tree()
            self.assertTrue(hasattr(r, 'intersection'))

            # some memory issues only show up when you copy the mesh a bunch
            for i in range(100):
                #c = mesh.copy()
                        
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
