import generic as g

class ThicknessTest(g.unittest.TestCase):

    def test_sphere_thickness(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)
        
        # check that thickness measures are non-negative
        thickness = g.trimesh.proximity.thickness(m, samples, exterior=False,
                                                  normals=m.face_normals[faces],
                                                  method='max_sphere')
        assert (thickness > -g.trimesh.tol.merge).all()
        
        # check thickness at a specific point
        point = g.np.array([[0.5, 0., 0.]])
        point_thickness = g.trimesh.proximity.thickness(m, point, 
                                                        exterior=False,
                                                        method='max_sphere')
        assert (g.np.abs(point_thickness - 0.5) <  g.trimesh.tol.merge)

    def test_ray_thickness(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)
        
        # check that thickness and measures are non-negative
        thickness = g.trimesh.proximity.thickness(m, samples, exterior=False,
                                                  normals=m.face_normals[faces],
                                                  method='ray')
        assert (thickness > -g.trimesh.tol.merge).all()
        
        # check thickness at a specific point
        point = g.np.array([[0.5, 0., 0.]])
        point_thickness = g.trimesh.proximity.thickness(m, point, 
                                                        exterior=False,
                                                        method='ray')
        assert (g.np.abs(point_thickness - 1.0) <  g.trimesh.tol.merge) 

    def test_sphere_reach(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)
        
        # check that reach measures are infinite
        reach = g.trimesh.proximity.thickness(m, samples, exterior=True,
                                              normals=m.face_normals[faces],
                                              method='max_sphere')
        assert g.np.isinf(reach).all()
        

    def test_ray_reach(self):
        m = g.trimesh.creation.box()

        samples, faces = m.sample(1000, return_index=True)
        
        # check that reach measures are infinite
        reach = g.trimesh.proximity.thickness(m, samples, exterior=True,
                                              normals=m.face_normals[faces],
                                              method='ray')
        assert g.np.isinf(reach).all()
        
        
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
