import gmsh, sys, os

def test_mesh3D():
    """

    """
    m = trimesh.creation.icosahedron()
    m.vertices,m.faces=trimesh.remesh.subdivide_to_size(m.vertices,m.faces,0.5)

    try:
        mesh3D(m,None,max_elm_size=None,mesher_id=1)
        assert True
    except:
        assert False
