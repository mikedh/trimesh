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
        dimension = (10000, 3)
        sphere = g.get_mesh('unit_sphere.STL', use_embree=False)

        ray_origins = g.np.random.random(dimension)
        ray_directions = g.np.tile([0, 0, 1], (dimension[0], 1))
        ray_origins[:, 2] = -5

        # force ray object to allocate tree before timing it
        #tree = sphere.ray.tree
        tic = g.time.time()
        sphere.ray.intersects_id(ray_origins, ray_directions)
        toc = g.time.time()
        rps = dimension[0] / (toc - tic)
        g.log.info('Measured %f rays/second', rps)

    def test_contains(self):
        mesh = g.get_mesh('unit_cube.STL')
        scale = 1.5

        test_on = mesh.contains(mesh.vertices)
        test_in = mesh.contains(mesh.vertices * (1.0 / scale))
        test_out = mesh.contains(mesh.vertices * scale)

        #assert test_on.all()
        self.assertTrue(test_in.all())
        self.assertFalse(test_out.any())

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()

    '''
    # sandbox to generate ray_data
    file_names = ['octagonal_pocket.ply',
                  'featuretype.STL',
                  'soup.stl',
                  'ballA.off']


    kwargs = [{'file_name' : f, 
               'use_embree' : e} for f,e in g.itertools.product(file_names,
                                                                [True, False])]
    meshes = [g.get_mesh(**k) for k in kwargs]
    names  = [i.metadata['file_name'] for i in meshes]
    rays = dict()

    # number or frays
    rl = 300

    # number of random vectors per origin
    nr = 3

    
    for m,name in zip(meshes, names):
        name = m.metadata['file_name']
        origins = g.trimesh.sample.volume_rectangular(
            extents=m.bounding_box.primitive.extents*3,
            count=rl*2,
            transform=m.bounding_box.primitive.transform)

        origins = origins[m.nearest.signed_distance(origins) < -.05][:rl]
        
        directions = g.np.column_stack((m.centroid - origins,
                                        g.np.random.random((len(origins),3*nr)))).reshape((-1,3))

        directions = g.trimesh.unitize(directions)
        
        forigins = g.np.tile(origins, nr+1).reshape((-1,3))

        rays[name] = {'ray_origins' : forigins.tolist(),
                      'ray_directions' : directions.tolist()}
    '''
