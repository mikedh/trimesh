try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import gmsh
except ImportError:
    pass

def test_generate():
    """

    """
    m = g.trimesh.creation.icosahedron()
    
    try:
        g.trimesh.solid_fea.generate(m)
        assert True
    except:
        assert False
