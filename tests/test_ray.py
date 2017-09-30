import generic as g


class RayTests(g.unittest.TestCase):

    def test_rays(self): 
        meshes = [g.get_mesh(**k) for k in g.data['ray_data']['load_kwargs']]
        rays = g.data['ray_data']['rays']
        names = [m.metadata['file_name'] for m in meshes]

        hit_id = []
        hit_loc = []
        hit_any = []
        for m in meshes:
            name = m.metadata['file_name']
            hit_any.append(m.ray.intersects_any(**rays[name]))
            hit_loc.append(len(m.ray.intersects_location(**rays[name])[0]))
            hit_id.append(m.ray.intersects_id(**rays[name]))
        hit_any = g.np.array(hit_any, dtype=g.np.int)

        for i in g.trimesh.grouping.group(g.np.unique(names, return_inverse=True)[1]):
            broken = hit_any[i].astype(g.np.int).ptp(axis=0).sum()
            self.assertTrue(broken == 0)

    def test_rps(self):
        for use_embree in [True, False]:
            dimension = (10000, 3)
            sphere = g.get_mesh('unit_sphere.STL',
                                use_embree=use_embree)

            ray_origins = g.np.random.random(dimension)
            ray_directions = g.np.tile([0, 0, 1], (dimension[0], 1))
            ray_origins[:, 2] = -5

            # force ray object to allocate tree before timing it
            #tree = sphere.ray.tree
            tic = [g.time.time()]
            sphere.ray.intersects_id(ray_origins, ray_directions)
            tic.append(g.time.time())
            sphere.ray.intersects_location(ray_origins, ray_directions)
            tic.append(g.time.time())
            
            rps = dimension[0] / g.np.diff(tic)

            g.log.info('Measured %s rays/second with embree %d',
                       str(rps),
                       use_embree)

    def test_contains(self):
        scale = 1.5
        for use_embree in [True, False]:
            mesh = g.get_mesh('unit_cube.STL', use_embree=use_embree)
            g.log.info('Contains test ray engine: ' + str(mesh.ray.__class__))
            
            test_on = mesh.ray.contains_points(mesh.vertices)
            test_in = mesh.ray.contains_points(mesh.vertices * (1.0 / scale))
            assert test_in.all()
            
            test_out = mesh.ray.contains_points(mesh.vertices * scale)
            assert not test_out.any()
            
            points_way_out = (g.np.random.random((30,3)) * 100) + 1.0 + mesh.bounds[1]
            test_way_out = mesh.ray.contains_points(points_way_out)
            assert not test_way_out.any()
            
            test_centroid = mesh.ray.contains_points([mesh.center_mass])
            assert test_centroid.all()
            

    def test_on_vertex(self):
        for use_embree in [True, False]:
            m = g.trimesh.primitives.Box(use_embree=False)

            origins = g.np.zeros_like(m.vertices)
            vectors = m.vertices.copy()

            assert m.ray.intersects_any(ray_origins=origins,
                                        ray_directions=vectors).all()

            (locations,
             index_ray,
             index_tri) = m.ray.intersects_location(ray_origins=origins,
                                                    ray_directions=vectors)

            hit_count = g.np.bincount(index_ray,
                                      minlength=len(origins))

            assert (hit_count == 1).all()

    def test_on_edge(self):
        for use_embree in [True, False]:
            m = g.get_mesh('7_8ths_cube.stl')
            
            points = [[4.5, 0, -23], [4.5, 0, -2], [0,0,-1e-6], [0,0,-1]]
            truth  = [False, True, True, True]
            result = g.trimesh.ray.ray_util.contains_points(m.ray, points)
            
            assert (result==truth).all()
            
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
