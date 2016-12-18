import generic as g

class RayTests(g.unittest.TestCase):
    def setUp(self):
        data = g.data['ray_data']
        self.meshes = [g.get_mesh(f) for f in data['filenames']]
        self.rays   = data['rays']
        self.truth  = data['truth']

    def test_rays(self):
        for mesh, ray_test, truth in zip(self.meshes, self.rays, self.truth):
            ray_test = g.np.array(ray_test)
            ray_origins = ray_test[:,0,:]
            ray_directions = ray_test[:,1,:]

            hit_id      = mesh.ray.intersects_id(ray_origins, 
                                                 ray_directions)
            hit_loc     = mesh.ray.intersects_location(ray_origins, 
                                                       ray_directions)
            hit_any     = mesh.ray.intersects_any(ray_origins, 
                                                  ray_directions)
            hit_any_tri = mesh.ray.intersects_any_triangle(ray_origins, 
                                                           ray_directions)

            for i in range(len(ray_origins)):
                self.assertTrue(len(hit_id[i])  == truth['count'][i])

    def test_rps(self):
        dimension = (1000,3)
        sphere    = g.get_mesh('unit_sphere.STL')

        ray_origins = g.np.random.random(dimension)
        ray_directions = g.np.tile([0,0,1], (dimension[0], 1))
        ray_origins[:,2] = -5
      
        # force ray object to allocate tree before timing it
        tree = sphere.triangles_tree()
        tic = g.time.time()
        sphere.ray.intersects_id(ray_origins, ray_directions)
        toc = g.time.time()
        rps = dimension[0] / (toc-tic)
        g.log.info('Measured %f rays/second', rps)

    def test_contains(self):
        mesh = g.get_mesh('unit_cube.STL')
        scale = 1+(g.trimesh.constants.tol.merge*2)

        test_on  = mesh.contains(mesh.vertices)
        test_in  = mesh.contains(mesh.vertices * (1.0/scale))
        test_out = mesh.contains(mesh.vertices * scale)
        
        #assert test_on.all()
        self.assertTrue(test_in.all())
        self.assertFalse(test_out.any())

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
