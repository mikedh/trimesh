import numpy as np

from .util import zero_pad
from .grouping import group_rows
from .constants import log, _log_time


_MIN_BIN_COUNT = 20
_TOL_FREQ = 1e-3


def rotationally_invariant_identifier(mesh, length=6):
    '''
    Given an input mesh, return a vector or string that has the following properties:
    * invariant to rotation of the mesh
    * robust to different tesselation of the surfaces
    * meshes that are similar but not identical return values that are close in euclidean distance

    Does this by computing the area- weighted distribution of the radius (from the center of mass).

    Arguments
    ---------
    mesh:    Trimesh
    length:  number of terms to compute of the identifier

    Returns
    ---------
    identifer: (length) float array of unique identifier
    '''

    frequency_count = int(length - 2)

    # calculate the mass properties of the mesh, which is doing a surface integral to
    # find the center of volume of the mesh
    mass_properties = mesh.mass_properties(skip_inertia=True)
    vertex_radii = np.sum((mesh.vertices.view(np.ndarray) - mesh.center_mass)**2,
                          axis=1) ** .5

    # since we will be computing the shape distribution of the radii, we need to make sure there
    # are enough values to populate more than one sample per bin.
    bin_count = int(np.min([256,
                            mesh.vertices.shape[0] * 0.2,
                            mesh.faces.shape[0] * 0.2]))

    # if any of the frequency checks fail, we will use this zero length vector as the
    # formatted information for the identifier
    freq_formatted = np.zeros(frequency_count)

    if bin_count > _MIN_BIN_COUNT:
        face_area = mesh.area_faces
        face_radii = vertex_radii[mesh.faces].reshape(-1)
        area_weight = np.tile(
            (face_area.reshape((-1, 1)) * (1.0 / 3.0)), (1, 3)).reshape(-1)

        if face_radii.std() > 1e-3:
            freq_formatted = fft_freq_histogram(face_radii,
                                                bin_count=bin_count,
                                                frequency_count=frequency_count,
                                                weight=area_weight)

    # using the volume (from surface integral), surface area, and top
    # frequencies
    identifier = np.hstack((mass_properties['volume'],
                            mass_properties['surface_area'],
                            freq_formatted))
    return identifier


def fft_freq_histogram(data, bin_count, frequency_count=4, weight=None):
    data = np.reshape(data, -1)
    if weight is None:
        weight = np.ones(len(data))

    hist, bin_edges = np.histogram(data,
                                   weights=weight,
                                   bins=bin_count)
    # we calculate the fft of the radius distribution
    fft = np.abs(np.fft.fft(hist))
    # the magnitude is dependant on our weighting being good
    # frequency should be more solid in more cases
    freq = np.fft.fftfreq(data.size, d=(
        bin_edges[1] - bin_edges[0])) + bin_edges[0]

    # now we must select the top FREQ_COUNT frequencies
    # if there are a bunch of frequencies whose components are very close in magnitude,
    # just picking the top FREQ_COUNT of them is non-deterministic
    # thus we take the top frequencies which have a magnitude that is distingushable
    # and we zero pad if this means fewer values available
    fft_top = fft.argsort()[-(frequency_count + 1):]
    fft_ok = np.diff(fft[fft_top]) > _TOL_FREQ
    if fft_ok.any():
        fft_start = np.nonzero(fft_ok)[0][0] + 1
        fft_top = fft_top[fft_start:]
        freq_final = np.sort(freq[fft_top])
    else:
        freq_final = []

    freq_formatted = zero_pad(freq_final, frequency_count)
    return freq_formatted


@_log_time
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
    # so we can use advanced indexing
    meshes = np.array(meshes)
    # by default an identifier is a 1D float array with 6 elements
    hashes = [i.identifier for i in meshes]
    groups = group_rows(hashes, digits=1)
    merged = [None] * len(groups)
    for i, group in enumerate(groups):
        quantity = 0
        metadata = {}
        for mesh in meshes[group]:
            # if metadata exists don't nuke it
            if 'quantity' in mesh.metadata:
                quantity += mesh.metadata['quantity']
            else:
                quantity += 1
            metadata.update(mesh.metadata)

        metadata['quantity'] = int(quantity)
        metadata['original_index'] = group

        merged[i] = meshes[group[0]]
        merged[i].metadata = metadata
    log.info('merge_duplicates reduced part count from %d to %d',
             len(meshes),
             len(merged))
    return np.array(merged)
