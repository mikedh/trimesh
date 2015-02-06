import numpy as np

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_FORMAT_STRING = '.6f'
_MIN_BIN_COUNT = 20
_FREQ_COUNT    = 6
_TOL_FREQ      = 1e-3

def _format_json(data):
    '''
    Built in json library doesn't have a good way of setting the precision of floating point
    numbers. Since we intend to use the string as a key in a dict, we need formatting to be
    identitical and well understood. 
    '''
    result = '[' + ','.join(map(lambda o: format(o, _FORMAT_STRING), data)) + ']'
    return result

def _zero_pad(data, count):
    '''
    Arguments
    --------
    data: (n) length 1D array 
    count: int

    Returns
    --------
    padded: (count) length 1D array if (n < count), otherwise length (n)
    '''
    if len(data) == 0:
        return np.zeros(count)
    elif len(data) < count:
        padded = np.zeros(count)
        padded[-len(data):] = data
        return padded
    else: return data

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
    bin_count = int(np.min([256, 
                            mesh.vertices.shape[0] * 0.2, 
                            mesh.faces.shape[0]    * 0.2]))
    
    # if any of the frequency checks fail, we will use this zero length vector as the 
    # formatted information for the identifier
    freq_formatted = np.zeros(_FREQ_COUNT)

    if bin_count > _MIN_BIN_COUNT:
        face_area       = mesh.area(sum=False)
        face_radii      = vertex_radii[mesh.faces]
        area_weight     = np.tile((face_area.reshape((-1,1))*(1.0/3.0)), (1,3))
        hist, bin_edges = np.histogram(face_radii.reshape(-1), bins=bin_count, weights=area_weight.reshape(-1))

        # we calculate the fft of the radius distribution
        fft  = np.abs(np.fft.fft(hist))
        # the magnitude is dependant on our area weighting being good, which it definitely isn't
        # frequency should be more solid
        freq = np.fft.fftfreq(face_radii.size, d=(bin_edges[1] - bin_edges[0]))

        # now we must select the top FREQ_COUNT frequencies
        # if there are a bunch of frequencies whose components are very close in magnitude,
        # just picking the top FREQ_COUNT of them is non-deterministic
        # thus we take the top frequencies which have a magnitude that is distingushable 
        # and we zero pad if this means fewer values available
        fft_top = fft.argsort()[-(_FREQ_COUNT + 1):] 
        fft_ok  = np.diff(fft[fft_top]) > _TOL_FREQ
        # only include freqeuncy information if they are distingushable above background noise
        if fft_ok.any():
            fft_start = np.nonzero(fft_ok)[0][0] + 1 
            fft_top   = fft_top[fft_start:]
            freq_formatted = _zero_pad(np.sort(freq[fft_top]), _FREQ_COUNT)
    else: 
        log.warn('Mesh isn\'t dense enough to calculate frequency information for unique identifier!')
        
    # using the volume (from surface integral), mean radius, and top frequencies
    identifier = np.hstack((mass_properties['volume'],
                            vertex_radii.mean(),
                            freq_formatted))

    # return as a json string rather than an array
    return _format_json(identifier)

if __name__ == '__main__':
    import trimesh
    import json
    from collections import deque
    mesh = trimesh.load_mesh('models/segway_wheel_left.STL')
    
    result = deque()
    for i in xrange(100):
        mesh.rezero()
        matrix = trimesh.transformations.random_rotation_matrix()
        matrix[0:3,3] = (np.random.random(3)-.5)*20
        mesh.transform(matrix)
        result.append(json.loads(rotationally_invariant_identifier(mesh)))

    ok = (np.abs(np.diff(result, axis=0)) < 1e-3).all()
    if not ok:
        print 'Hashes differ after transform! diffs:\n %s\n' % str(np.diff(result, axis=0))
