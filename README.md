trimesh.py
==========

Simple, single-file python library for loading triangular meshes (STL in particular):

    import os
    import transformations
    m = load(os.path.join('./models', 'round.stl'))
    tr = transformations.rotation_matrix(np.radians(34), [1,0,0])
    m.transform(tr)