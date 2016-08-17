import generic as g

class NormalsTest(g.unittest.TestCase):
    def test_vertex_normal(self):
        mesh = g.trimesh.creation.icosahedron()
        # the icosahedron is centered at zero, so the true vertex
        # normal is just a unit vector of the vertex position
        truth = g.trimesh.util.unitize(mesh.vertices)

        # force fallback to loop normal summing by passing None as the sparse matrix
        normals = g.trimesh.geometry.mean_vertex_normals(len(mesh.vertices), 
                                                         mesh.faces, 
                                                         mesh.face_normals, 
                                                         sparse=None)
        self.assertTrue(g.np.allclose(normals-truth, 0.0))

        # make sure the automatic sparse matrix generation works as well
        normals = g.trimesh.geometry.mean_vertex_normals(len(mesh.vertices), 
                                                         mesh.faces, 
                                                         mesh.face_normals)
        self.assertTrue(g.np.allclose(normals-truth, 0.0))
        
        # make sure the Trimesh normals- related attributes are wired correctly
        self.assertTrue(mesh.faces_sparse is not None)
        self.assertTrue(mesh.vertex_normals.shape == mesh.vertices.shape)
        self.assertTrue(g.np.allclose(mesh.vertex_normals-truth, 0.0))
        
    def test_face_normals(self):
        mesh = g.trimesh.creation.icosahedron()
        self.assertTrue(mesh.face_normals.shape == mesh.faces.shape)
        mesh.face_normals = None
        self.assertTrue(mesh.face_normals.shape == mesh.faces.shape)
        
if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
    
