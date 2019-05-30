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
    sx, sy, sz = (int(s) for s in shape)
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


def voxel_from_binvox(
        rle_data, shape, translate=None, scale=1.0, axis_order='xzy'):
    """
    Factory for building from data associated with binvox files.

    Args:
        rle_data: numpy array representing run-length-encoded of flat voxel
            values, or a `trimesh.rle.RunLengthEncoding` object.
            See `trimesh.rle` documentation for description of encoding.
        shape: shape of voxel grid.
        translate: alias for `origin` in trimesh terminology
        scale: side length of entire voxel grid. Note this is different
            to `pitch` in trimesh terminology, which relates to the side
            length of an individual voxel.
        encoded_axes: iterable with values in ('x', 'y', 'z', 0, 1, 2),
            where x => 0, y => 1, z => 2
            denoting the order of axes in the encoded data. binvox by
            default saves in xzy order, but using `xyz` (or (0, 1, 2)) will
            be faster in some circumstances.

    Returns:
        `Voxel` instance
    """
    # shape must be uniform else scale is ambiguous
    from ..voxel import encoding as enc
    from .. import voxel as v
    from .. import transformations
    if isinstance(rle_data, enc.RunLengthEncoding):
        encoding = rle_data
    else:
        encoding = enc.RunLengthEncoding(rle_data, dtype=bool)

    # translate = np.asanyarray(translate) * scale)
    # translate = [0, 0, 0]
    transform = transformations.scale_and_translate(
        scale=scale / (np.array(shape) - 1),
        translate=translate)

    if axis_order == 'xzy':
        perm = (0, 2, 1)
        shape = tuple(shape[p] for p in perm)
        encoding = encoding.reshape(shape).transpose(perm)
    elif axis_order is None or axis_order == 'xyz':
        encoding = encoding.reshape(shape)
    else:
        raise ValueError(
            "Invalid axis_order '%s': must be None, 'xyz' or 'xzy'")

    assert(encoding.shape == shape)
    return v.Voxel(encoding, transform)


def load_binvox(file_obj, resolver=None, axis_order='xzy', **kwargs):
    """Load trimesh `Voxel` instance from file.

    Args:
        file_obj: file-like object with `read` and `readline` methods.
        resolve: unused
        axis_order: order of axes in encoded data. binvox default is
            'xzy', but 'xyz' may be faster results where this is not relevant.
        **kwargs: unused

    Returns:
        `trimesh.voxel.VoxelBase` instance.
    """
    data = parse_binvox(file_obj)
    return voxel_from_binvox(
        rle_data=data.rle_data,
        shape=data.shape,
        translate=data.translate,
        scale=data.scale,
        axis_order=axis_order)


def export_binvox(voxel, axis_order='xzy'):
    """Export `trimesh.voxel.VoxelBase` instance to bytes

    Args:
        voxel: `trimesh.voxel.VoxelBase` instance. Assumes axis ordering of
            `xyz` and encodes in binvox default `xzy` ordering.
        axis_order: iterable of elements in ('x', 'y', 'z', 0, 1, 2), the order
            of axes to encode data (standard is 'xzy' for binvox). `voxel`
            data is assumed to be in order 'xyz'.

    Returns:
        bytes representation according to binvox spec
    """
    transform = voxel.transform
    translate = transform.matrix[:3, 3]
    encoding = voxel.encoding

    tol = 1e-12
    i, j = np.where(np.abs(transform.matrix[:3, :3]) > tol)
    scales = transform.matrix[i, j] * (np.array(voxel.shape) - 1)

    if not np.all(i == j) or np.any(scales) < 0:
        # TODO: refactor transpose/reflection in transform into
        # transpose/flip in encoding
        raise ValueError(
            'Invalid transformation matrix for exporting to binvox - '
            'no rotation/shear/reflection allowed.')

    if not np.all(np.abs(scales[1:] - scales[0]) < tol):
        raise ValueError(
            'Invalid transformation matrix for exporting to binvox - '
            'only uniform scales allowed')
    scale = scales[0]
    if axis_order == 'xzy':
        encoding = encoding.transpose((0, 2, 1))
    elif axis_order != 'xyz':
        raise ValueError('Invalid axis_order: must be one of ("xyz", "xzy")')
    rle_data = encoding.flat.run_length_data(dtype=np.uint8)
    return binvox_bytes(
        rle_data, shape=voxel.shape, translate=translate, scale=scale)
