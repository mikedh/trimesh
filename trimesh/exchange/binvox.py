"""Parsing functions for Binvox files.

https://www.patrickmin.com/binvox/binvox.html
"""

import numpy as np
import collections
Binvox = collections.namedtuple(
    'Binvox', ['rle_data', 'shape', 'translate', 'scale'])


def parse_binvox_header(fp):
    """Read the header from a binvox file.

    Spec at https://www.patrickmin.com/binvox/binvox.html

    Args:
        fp: object providing a `readline` method (e.g. an open file)

    Returns:
        (shape, translate, scale) according to binvox spec

    Raises:
        `IOError` if invalid binvox file.
    """

    line = fp.readline().strip()
    if hasattr(line, 'decode'):
        binvox = b'#binvox'
        space = b' '
    else:
        binvox = '#binvox'
        space = ' '
    if not line.startswith(binvox):
        raise IOError('Not a binvox file')
    shape = tuple(
        int(s) for s in fp.readline().strip().split(space)[1:])
    translate = tuple(
        float(s) for s in fp.readline().strip().split(space)[1:])
    scale = float(fp.readline().strip().split(space)[1])
    fp.readline()
    return shape, translate, scale


def parse_binvox(fp):
    """Read a binvox file.

    Spec at https://www.patrickmin.com/binvox/binvox.html

    Args:
        fp: object providing a `readline` method (e.g. an open file)

    Returns:
        `Binvox` namedtuple ('rle', 'shape', 'translate', 'scale')
        `rle` is the run length encoding of the values.

    Raises:
        `IOError` if invalid binvox file.
    """
    shape, translate, scale = parse_binvox_header(fp)
    data = fp.read()
    if hasattr(data, 'encode'):
        data = data.encode()
    rle_data = np.frombuffer(data, dtype=np.uint8)
    return Binvox(rle_data, shape, translate, scale)


_binvox_header = '''#binvox 1
dim {sx} {sy} {sz}
translate {tx} {ty} {tz}
scale {scale}
data
'''

def binvox_header(shape, translate, scale):
    """Get a binvox header string.

    Args:
        shape: length 3 iterable of ints denoting shape of voxel grid.
        translate: length 3 iterable of floats denoting translation.
        scale: num length of entire voxel grid.

    Returns:
        string including "data\n" line.
    """
    sx, sy, sz = shape
    if not all(isinstance(s, int) for s in shape):
        raise ValueError('All shape elements must be ints')
    tx, ty, tz = translate
    return _binvox_header.format(
        sx=sx, sy=sy, sz=sz, tx=tx, ty=ty, tz=tz, scale=scale)


def binvox_bytes(rle_data, shape, translate=(0, 0, 0), scale=1):
    """Get a binary representation of binvoxe data.

    Args:
        rle_data: run-length encoded numpy array.
        shape: length 3 iterable of ints denoting shape of voxel grid.
        translate: length 3 iterable of floats denoting translation.
        scale: num length of entire voxel grid.

    Returns:
        bytes representation, suitable for writing to binary file
    """
    if rle_data.dtype != np.uint8:
        raise ValueError(
            "rle_data.dtype must be np.uint8, got %s" % rle_data.dtype)

    header = binvox_header(shape, translate, scale).encode()
    return header + rle_data.tostring()


def load_binvox(file_obj, resolver=None, encoded_axes='xzy', **kwargs):
    """Load trimesh `Voxel` instance from file.

    Args:
        file_obj: file-like object with `read` and `readline` methods.
        resolve: unused
        encoded_axes: order of axes in encoded data. binvox default is
            'xzy', but 'xyz' may be faster results where this is not relevant.
        **kwargs: unused

    Returns:
        `trimesh.voxel.VoxelBase` instance.
    """
    from .. import voxel
    data = parse_binvox(file_obj)
    return voxel.VoxelRle.from_binvox_data(
        rle_data=data.rle_data,
        shape=data.shape,
        translate=data.translate,
        scale=data.scale,
        encoded_axes=encoded_axes)


def export_binvox(voxel, encoded_axes='xzy'):
    """Export `trimesh.voxel.VoxelBase` instance to bytes

    Args:
        voxel: `trimesh.voxel.VoxelBase` instance. Assumes axis ordering of
            `xyz` and encodes in binvox default `xzy` ordering.
        encoded_axes: iterable of elements in ('x', 'y', 'z', 0, 1, 2)

    Returns:
        bytes representation according to binvox spec
    """
    indices = {'x': 0, 'y': 1, 'z': 2}
    axes = tuple(indices.get(a, a) for a in encoded_axes)
    if not (len(axes) == 3 and all(i in axes for i in range(3))):
        raise ValueError('Invalid encoded_axes %s' % (encoded_axes))
    voxel = voxel.transpose(axes)
    rle_data = getattr(voxel, 'rle_data', None)
    if rle_data is None:
        from .. import rle
        rle_data = rle.dense_to_rle(voxel.matrix, dtype=np.uint8)
    shape = voxel.shape
    if not (shape[0] == shape[1] == shape[2]):
        raise ValueError(
            'trimesh only supports uniform scaling, so required binvox '
            'with uniform shapes')
    return binvox_bytes(
        rle_data, shape=voxel.shape, translate=voxel.origin,
        scale=voxel.pitch * (shape[0] - 1))
