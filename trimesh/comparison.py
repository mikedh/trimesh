import numpy as np

from .grouping import group_rows
from .constants import log_time
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_MIN_BIN_COUNT = 20
_TOL_FREQ      = 1e-3

def _format_json(data, digits=6):
    '''
    Built in json library doesn't have a good way of setting the precision of floating point
    numbers. Since we intend to use the string as a key in a dict, we need formatting to be
    identitical and well understood. 
    '''
    format_str = '.' + str(int(digits)) + 'f'
    result     = '[' + ','.join(map(lambda o: format(o, format_str), data)) + ']'
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

@log_time
def merge_duplicates(meshes):
    '''
    Given a list of meshes, find meshes which are duplicates and merge them.

    Arguments
    ---------
    meshes: (n) list of meshes

    Returns
    ---------
    merged: (m) list of meshes where (m <= n)
    '''
    hashes = [i.identifier() for i in meshes]
    groups = group_rows(hashes, digits=1)
    merged = [None] * len(groups)
    for i, group in enumerate(groups):
        merged[i] = meshes[group[0]]
        merged[i].metadata['quantity'] = len(group)
        
    log.info('merge_duplicates reduced part count from %d to %d', 
             len(meshes),
             len(merged))
    return np.array(merged)

def rotationally_invariant_identifier(mesh, length=6, as_json=False):
    '''
    Given an input mesh, return a vector or string that has the following properties:
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

    frequency_count = int(length - 2)

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
    freq_formatted = np.zeros(frequency_count)

    if bin_count > _MIN_BIN_COUNT:
        face_area       = mesh.area(sum=False)
        face_radii      = vertex_radii[mesh.faces]
        area_weight     = np.tile((face_area.reshape((-1,1))*(1.0/3.0)), (1,3))
        hist, bin_edges = np.histogram(face_radii.reshape(-1), 
                                       bins=bin_count, 
                                       weights=area_weight.reshape(-1))

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
        fft_top = fft.argsort()[-(frequency_count + 1):] 
        fft_ok  = np.diff(fft[fft_top]) > _TOL_FREQ
        # only include freqeuncy information if they are distingushable above background noise
        if fft_ok.any():
            fft_start = np.nonzero(fft_ok)[0][0] + 1 
            fft_top   = fft_top[fft_start:]
            freq_formatted = _zero_pad(np.sort(freq[fft_top]), frequency_count)
    else: 
        log.debug('Mesh isn\'t dense enough to calculate frequency information for unique identifier!')
        
    # using the volume (from surface integral), surface area, and top frequencies
    identifier = np.hstack((mass_properties['volume'],
                            mass_properties['surface_area'],
                            freq_formatted))
    if as_json:
        # return as a json string rather than an array
        return _format_json(identifier)
    return identifier

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
