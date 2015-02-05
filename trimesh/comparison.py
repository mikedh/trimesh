import numpy as np
_FORMAT_STRING = '.6f'

def _format_json(data):
    '''
    Built in json library doesn't have a good way of setting the precision of floating point
    numbers. Since we intend to use the string as a key in a dict, we need formatting to be
    identitical and well understood. 
    '''
    result = '[' + ','.join(map(lambda o: format(o, _FORMAT_STRING), data)) + ']'
    return result

def rotationally_invariant_identifier(mesh):
    '''
    Given an input mesh, return a string that has the following properties:
    * invariant to rotation of the mesh
    * robust to different tesselation of the surfaces
    * meshes that are similar but not identical return values that are close in euclidean distance

    Does this by computing the area- weighted distribution of the radius (from the center of mass).

    Arguments
    ---------
    mesh: Trimesh

    Returns
    ---------
    identifer: string representing mesh. Formatting is 1D array in JSON. 
    '''

    # calculate the mass properties of the mesh, which is doing a surface integral to
    # find the center of volume of the mesh
    mass_properties = mesh.mass_properties(skip_inertia=True)
    center_mass     = mass_properties['center_mass']
    vertex_radii    = np.sum((mesh.vertices - mesh.center_mass)**2, axis=1) **.5
    
    # since we will be computing the shape distribution of the radii, we need to make sure there
    # are enough values to populate more than one sample per bin.  
    bin_count = np.min([256, 
                        mesh.vertices.shape[0] * 0.25, 
                        mesh.faces.shape[0]    * 0.25])

    face_area       = mesh.area(sum=False)
    face_radii      = vertex_radii[mesh.faces]
    area_weight     =  np.tile((face_area.reshape((-1,1))*(1.0/3.0)), (1,3))
    hist, bin_edges = np.histogram(face_radii.reshape(-1), bins=bin_count, weights=area_weight.reshape(-1))

    # we calculate the fft of the radius distribution
    fft  = np.abs(np.fft.fft(hist))
    # the magnitude is dependant on our area weighting being good, which it definitely isn't
    # frequency should be more solid
    freq = np.fft.fftfreq(face_radii.size, d=(bin_edges[1] - bin_edges[0]))
    
    # using the volume (from surface integral), mean radius, and top frequencies
    # note that we are sorting the top 5 frequencies here. 
    identifier = np.hstack((mass_properties['volume'],
                            vertex_radii.mean(),
                            np.sort(freq[np.argsort(fft)[-5:]])))

    # return as a json string rather than an array
    return _format_json(identifier)


if __name__ == '__main__':
    import trimesh
    m = trimesh.load_mesh('models/featuretype.STL')
    #m = trimesh.load_mesh('models/nonconvex.STL')
    #m = trimesh.load_mesh('models/ballA.off')

    from collections import deque
    import json
    np.set_printoptions(precision=5, suppress=True)

    result = deque()
    for i in xrange(100):
        matrix = trimesh.transformations.random_rotation_matrix()
        matrix[0:3,3] = (np.random.random(3)-.5)*20
        m.transform(matrix)
        result.append(json.loads(rotationally_invariant_identifier(m)))

    ok = (np.abs(np.diff(result, axis=0)) < 1e-8).all()
    print np.diff(result, axis=0)
    print ok
