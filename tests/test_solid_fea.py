try:
    import gmsh
except ImportError:
    pass

def test_generate():
    """

    """
    m = trimesh.creation.icosahedron()
    
    try:
        generate(m)
        assert True
    except:
        assert False
