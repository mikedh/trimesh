"""
voxel.py
-----------

Convert meshes to a simple voxel data structure and back again.
"""
import numpy as np

from .. import caching
from . import ops
from ..constants import log
from ..parent import Geometry
from .. import transformations as tr
from . import encoding as enc
from .. import util
from . import morphology


class _Transform(object):
    """Class for caching metadata associated with 4x4 transformations."""

    def __init__(self, matrix):
        if matrix.shape != (4, 4):
            raise ValueError('matrix must be 4x4, got %s' % str(matrix.shape))
        if not np.all(matrix[3, :] == [0, 0, 0, 1]):
            raise ValueError('matrix not a valid transformation matrix')
        self._data = caching.tracked_array(matrix, dtype=float)
        self._cache = caching.Cache(id_function=self._data.crc)

    def md5(self):
        return self._data.md5()

    def crc(self):
        return self._data.crc()

    @property
    def translation(self):
        return self._data[:3, 3]

    @property
    def matrix(self):
        return self._data

    @matrix.setter
    def matrix(self, data):
        np.copyto(self._data, data.astype(float))

    @caching.cache_decorator
    def scale(self):
        matrix = self.matrix
        scale = np.diag(matrix[:3, :3])
        if not util.allclose(
                matrix[:3, :3], scale * np.eye(3), scale * 1e-6 + 1e-8):
            raise RuntimeError(
                'scale ill-defined because transform_matrix features '
                'a shear or rotation')
        return scale

    @caching.cache_decorator
    def pitch(self):
        scale = self.scale
        if not util.allclose(
                scale[0], scale[1:], np.max(np.abs(scale)) * 1e-6 + 1e-8):
            raise RuntimeError(
                'pitch ill-defined because transform_matrix features '
                'non-uniform scaling.')

    @caching.cache_decorator
    def unit_volume(self):
        """Volume of a transformed unit cube."""
        return np.linalg.det(self._data[:3, :3])

    def apply_transform(self, matrix):
        """Mutate the transform in-place and return self."""
        self.matrix = np.matmul(matrix, self.matrix)
        return self

    def apply_translation(self, translation):
        """Mutate the transform in-place and return self."""
        self.matrix[:3, 3] += translation
        return self

    def apply_scale(self, scale):
        """Mutate the transform in-place and return self."""
        self.matrix[:3] *= scale
        return self

    def transform_points(self, points):
        """
        Apply the transformation to points (not in-place).

        Parameters
        ----------
        points: (..., 3) float coordinates

        Returns
        ----------
        transformed points of the same shape as points, or the original points
        (i.e. not coppied) if this is an identity transform.
        """
        if self.is_identity:
            return points
        return tr.transform_points(
            points.reshape(-1, 3), self.matrix).reshape(points.shape)

    def inverse_transform_points(self, points):
        """Apply the inverse transformation to points (not in-place)."""
        if self.is_identity:
            return points
        return tr.transform_points(
            points.reshape(-1, 3), self.inverse_matrix).reshape(points.shape)

    @caching.cache_decorator
    def inverse_matrix(self):
        inv = np.linalg.inv(self.matrix)
        inv.flags.writeable = False
        return inv

    def copy(self):
        return _Transform(self._data.copy())

    @caching.cache_decorator
    def is_identity(self):
        """Flags this transformation being sufficiently close to eye(4)."""
        return util.allclose(self.matrix, np.eye(4), 1e-8)


class VoxelGrid(Geometry):
    def __init__(self, encoding, transform_matrix=None, metadata=None):
        if transform_matrix is None:
            transform_matrix = np.eye(4)
        if isinstance(encoding, np.ndarray):
            encoding = enc.DenseEncoding(encoding.astype(bool))
        if encoding.dtype != bool:
            raise ValueError('encoding must have dtype bool')
        self._data = caching.DataStore()
        self.encoding = encoding
        self._data['transform'] = _Transform(transform_matrix)
        self._cache = caching.Cache(id_function=self._data.crc)

        self.metadata = dict()
        # update the mesh metadata with passed metadata
        if isinstance(metadata, dict):
            self.metadata.update(metadata)
        elif metadata is not None:
            raise ValueError(
                'metadata should be a dict or None, got %s' % str(metadata))

    def md5(self):
        return self._data.md5()

    def crc(self):
        return self._data.crc()

    @property
    def encoding(self):
        """
        `Encoding` object providing the occupancy grid.

        See `trimesh.voxel.encoding` for implementations.
        """
        return self._data['encoding']

    @encoding.setter
    def encoding(self, encoding):
        if isinstance(encoding, np.ndarray):
            encoding = enc.DenseEncoding(encoding)
        elif not isinstance(encoding, enc.Encoding):
            raise ValueError(
                'encoding must be an Encoding, got %s' % str(encoding))
        if len(encoding.shape) != 3:
            raise ValueError(
                'encoding must be rank 3, got shape %s' % str(encoding.shape))
        if encoding.dtype != bool:
            raise ValueError(
                'encoding must be binary, got %s' % encoding.dtype)
        self._data['encoding'] = encoding

    @property
    def _transform(self):
        return self._data['transform']

    @property
    def transform_matrix(self):
        """4x4 homogeneous transformation matrix."""
        return self._transform.matrix

    @transform_matrix.setter
    def transform_matrix(self, matrix):
        """4x4 homogeneous transformation matrix."""
        self._transform.matrix = matrix

    @property
    def translation(self):
        """Location of voxel at [0, 0, 0]."""
        return self._transform.translation

    @property
    def origin(self):
        """Deprecated. Use `self.translation`."""
        # DEPRECATED. Use translation instead
        return self.translation

    @property
    def scale(self):
        """
        3-element float representing per-axis scale.

        Raises a `RuntimeError` if `self.transform_matrix` has rotation or
        shear components.
        """
        return self._transform.scale

    @property
    def pitch(self):
        """
        Uniform scaling factor representing the side length of each voxel.

        Raises a `RuntimeError` if `self.transformation` has rotation or shear
        components of has non-uniform scaling.
        """
        return self._transform.pitch

    @property
    def element_volume(self):
        return self._transform.unit_volume

    def apply_transform(self, matrix):
        self._transform.apply_transform(matrix)
        return self

    def strip(self):
        """
        Mutate self by stripping leading/trailing planes of zeros.

        Returns
        --------
        self after mutation occurs in-place
        """
        encoding, padding = self.encoding.stripped
        self.encoding = encoding
        self._transform.matrix[:3, 3] = self.indices_to_points(padding[:, 0])
        return self

    @caching.cache_decorator
    def bounds(self):
        points = self.points
        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        bounds.flags.writeable = False
        return bounds

    @caching.cache_decorator
    def extents(self):
        bounds = self.bounds
        extents = bounds[1] - bounds[0]
        extents.flags.writeable = False
        return extents

    @caching.cache_decorator
    def is_empty(self):
        return self.encoding.is_empty

    @property
    def shape(self):
        """3-tuple of ints denoting shape of occupancy grid."""
        return self.encoding.shape

    @caching.cache_decorator
    def filled_count(self):
        """int, number of occupied voxels in the grid."""
        return self.encoding.sum.item()

    def is_filled(self, point):
        """
        Query points to see if the voxel cells they lie in are filled or not.

        Parameters
        ----------
        point: (..., 3) float, point(s) in space

        Returns
        ---------
        is_filled: (...,) bool, is cell occupied or not for each point
        """
        point = np.asanyarray(point)
        indices = self.points_to_indices(point)
        in_range = np.logical_and(
            np.all(indices < np.array(self.shape), axis=-1),
            np.all(indices >= 0, axis=-1))

        is_filled = np.zeros_like(in_range)
        is_filled[in_range] = self.encoding.gather_nd(indices[in_range])
        return is_filled

    def fill(self, method='holes', **kwargs):
        """
        Mutates self by filling in the encoding according to `morphology.fill`.

        Parameters
        ----------
        method: implementation key, one of
            `trimesh.voxel.morphology.fill.fillers` keys
        **kwargs: additional kwargs passed to the keyed implementation

        Returns
        ----------
        self after replacing encoding with a filled version.
        """
        self.encoding = morphology.fill(self.encoding, method=method, **kwargs)
        return self

    def hollow(self, structure=None):
        """
        Mutates self by removing internal voxels leaving only surface elements.

        Surviving elements are those in encoding that are adjacent to an empty
        voxel, where adjacency is controlled by `structure`.

        Parameters
        ----------
        structure: adjacency structure. If None, square connectivity is used.

        Returns
        ----------
        self after replacing encoding with a surface version.
        """
        self.encoding = morphology.surface(self.encoding)
        return self

    @caching.cache_decorator
    def marching_cubes(self):
        """
        A marching cubes Trimesh representation of the voxels.

        No effort was made to clean or smooth the result in any way;
        it is merely the result of applying the scikit-image
        measure.marching_cubes function to self.encoding.dense.

        Returns
        ---------
        meshed: Trimesh object representing the current voxel
                        object, as returned by marching cubes algorithm.
        """
        meshed = ops.matrix_to_marching_cubes(matrix=self.encoding.dense)
        return meshed

    @caching.cache_decorator
    def volume(self):
        """
        What is the volume of the filled cells in the current voxel object.

        Returns
        ---------
        volume: float, volume of filled cells
        """
        return self.filled_count * self.element_volume

    @caching.cache_decorator
    def points(self):
        """
        The center of each filled cell as a list of points.

        Returns
        ----------
        points: (self.filled, 3) float, list of points
        """
        return self._transform.transform_points(
            self.sparse_indices.astype(float))

    @property
    def sparse_indices(self):
        """(n, 3) int array of sparse indices of occupied voxels."""
        return self.encoding.sparse_indices

    def as_boxes(self, colors=None):
        """
        A rough Trimesh representation of the voxels with a box
        for each filled voxel.

        Parameters
        ----------
        colors : (3,) or (4,) float or uint8
                 (X, Y, Z, 3) or (X, Y, Z, 4) float or uint8
         Where matrix.shape == (X, Y, Z)

        Returns
        ---------
        mesh : trimesh.Trimesh
          Mesh with one box per filled cell.
        """

        if colors is not None:
            colors = np.asanyarray(colors)
            if colors.ndim == 4:
                encoding = self.encoding
                if colors.shape[:3] == encoding.shape:
                    # TODO jackd: more efficient implementation?
                    # encoding.as_mask?
                    colors = colors[encoding.dense]
                else:
                    log.warning('colors incorrect shape!')
                    colors = None
            elif colors.shape not in ((3,), (4,)):
                log.warning('colors incorrect shape!')
                colors = None

        mesh = ops.multibox(
            centers=self.sparse_indices.astype(float), colors=colors)

        mesh = mesh.apply_transform(self.transform_matrix)
        return mesh

    def points_to_indices(self, points):
        """
        Convert points to indices in the matrix array.

        Parameters
        ----------
        points: (..., 3) float, point in space

        Returns
        ---------
        indices: (..., 3) int array of indices into self.encoding
        """
        points = self._transform.inverse_transform_points(points)
        return np.round(points).astype(int)

    def indices_to_points(self, indices):
        return self._transform.transform_points(indices.astype(float))

    def show(self, *args, **kwargs):
        """
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        """
        return self.as_boxes(kwargs.pop('colors', None)).show(*args, **kwargs)

    def copy(self):
        return VoxelGrid(self.encoding.copy(), self._transform.matrix.copy())

    def revoxelized(self, shape):
        """
        Create a new VoxelGrid without rotations, reflections or shearing.

        Parameters
        ----------
        shape: 3-tuple of ints denoting the shape of the returned VoxelGrid.

        Returns
        ----------
        VoxelGrid of the given shape with (possibly non-uniform) scale and
        translation transformation matrix.
        """
        from .. import util
        shape = tuple(shape)
        bounds = self.bounds.copy()
        extents = self.extents
        points = util.grid_linspace(bounds, shape).reshape(shape + (3,))
        dense = self.is_filled(points)
        scale = extents / np.asanyarray(shape)
        translate = bounds[0]
        return VoxelGrid(
            dense,
            transform_matrix=tr.scale_and_translate(scale, translate)
        )
