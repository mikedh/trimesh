import generic as g

class RayTests(g.unittest.TestCase):
    def setUp(self):
        data = g.data['ray_data']
        self.meshes = [g.get_mesh(f) for f in data['filenames']]
        self.rays   = data['rays']
        self.truth  = data['truth']

    def test_rays(self):
        for mesh, ray_test, truth in zip(self.meshes, self.rays, self.truth):
            hit_id      = mesh.ray.intersects_id(ray_test)
            hit_loc     = mesh.ray.intersects_location(ray_test)
            hit_any     = mesh.ray.intersects_any(ray_test)
            hit_any_tri = mesh.ray.intersects_any_triangle(ray_test)

            for i in range(len(ray_test)):
                self.assertTrue(len(hit_id[i])  == truth['count'][i])
                #self.assertTrue(len(hit_loc[i]) == truth['count'][i])

    def test_rps(self):
        dimension = (1000,3)
        sphere    = g.get_mesh('unit_sphere.STL')

        rays_ori = g.np.random.random(dimension)
        rays_dir = g.np.tile([0,0,1], (dimension[0], 1))
        rays_ori[:,2] = -5
        rays = g.np.column_stack((rays_ori, rays_dir)).reshape((-1,2,3))
        # force rayr object to allocate tree before timing it
        tree = sphere.triangles_tree()
        tic = g.time.time()
        sphere.ray.intersects_id(rays)
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
    
