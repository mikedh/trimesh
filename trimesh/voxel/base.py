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


# def scale_and_translate(scale=None, translate=None):
#     transform = IDENTITY_TRANSFORM
#     if scale is not None:
#         transform = transform.apply_scale(scale)
#     if translate is not None:
#         transform = transform.apply_translation(translate)
#     return transform


class Transformation(object):
    def __init__(self, matrix):
        self._data = caching.tracked_array(matrix)
        if not (matrix.shape == (4, 4) and np.all(matrix[3] == (0, 0, 0, 1))):
            raise ValueError(
                'matrix is not a valid homogeneous transformation matrix')
        self._cache = caching.Cache(id_function=self._data.crc)
        self._inverse = None

    def crc(self):
        return self._data.crc()

    def md5(self):
        return self._data.md5()

    @caching.cache_decorator
    def volume(self):
        return np.linalg.det(self.matrix[:3, :3])

    def transform_points(self, points):
        return tr.transform_points(
            points.reshape(-1, 3), self._data).reshape(points.shape)

    @property
    def translation(self):
        return self.matrix[:3, 3]

    def apply_transform(self, matrix):
        return Transformation.from_matrix(np.matmul(matrix, self.matrix))

    def apply_translation(self, translation):
        if np.all(np.asanyarray(translation) == 0):
            return self
        matrix = self.matrix.copy()
        matrix[:3, 3] += translation
        return Transformation.from_matrix(matrix)

    def apply_scale(self, scale):
        if np.all(np.asanyarray(scale) == 1):
            return self
        matrix = self.matrix.copy()
        matrix[:3] *= np.expand_dims(scale, axis=-1)
        return Transformation.from_matrix(matrix)

    @property
    def inverse(self):
        if self._inverse is None:
            self._inverse = InverseTransformation(np.linalg.inv(self.matrix))
        return self._inverse

    @property
    def matrix(self):
        return self._data

    @staticmethod
    def from_translation(translation):
        if np.all(np.asanyarray(translation) == 0):
            return IDENTITY_TRANSFORM
        return Transformation(tr.translation_matrix(translation))

    @staticmethod
    def from_scale(scale):
        if np.all(np.asanyarray(scale) == 1):
            return IDENTITY_TRANSFORM
        return Transformation(tr.scale_matrix(scale))

    @staticmethod
    def from_matrix(matrix):
        matrix = np.asanyarray(matrix)
        if matrix.shape == (3, 3):
            actual = np.zeros((4, 4))
            actual[:3, :3] = matrix
            actual[3, 3] = 1
            matrix = actual
        if np.all(matrix == _eye):
            return IDENTITY_TRANSFORM
        return Transformation(matrix)

    def copy(self):
        return Transformation(self._data.copy())


class InverseTransformation(Transformation):
    def __init__(self, base):
        self._data = base
        self._cache = caching.Cache(id_function=base.crc)

    def crc(self):
        return self._data.crc()

    def md5(self):
        return self._data.md5()

    @caching.cache_decorator
    def matrix(self):
        return np.linalg.inv(self._data.matrix)

    @property
    def inverse(self):
        return self._data

    @caching.cache_decorator
    def volume(self):
        return 1. / self.inverse.volume

    def copy(self):
        return self._data.copy().inverse


class _IdentityTransformation(Transformation):
    def __init__(self):
        pass

    def crc(self):
        return _eye_crc

    def md5(self):
        return _eye_md5

    @property
    def matrix(self):
        return _eye

    @property
    def volume(self):
        return 1.0

    def transform_points(self, points):
        return points

    def apply_transform(self, transform):
        return transform

    def apply_translation(self, translation):
        return Transformation.from_translation(translation)

    def apply_scale(self, scale):
        return Transformation.from_scale(scale)

    @property
    def inverse(self):
        return self

    def copy(self):
        return self


_eye = np.eye(4)
_eye.flags.writeable = False
_eye = caching.tracked_array(_eye)
_eye_crc = _eye.crc()
_eye_md5 = _eye.md5()
IDENTITY_TRANSFORM = _IdentityTransformation()


class Voxel(Geometry):
    def __init__(self, encoding, transform=IDENTITY_TRANSFORM):
        if isinstance(encoding, np.ndarray):
            encoding = enc.DenseEncoding(encoding.astype(bool))
        if encoding.dtype != bool:
            raise ValueError('encoding must have dtype bool')
        self._data = caching.DataStore()
        self._data['encoding'] = encoding
        self._cache = caching.Cache(id_function=self._data.crc)
        self.transform = transform

    def md5(self):
        return self._data.md5()

    def crc(self):
        return self._data.crc()

    @property
    def encoding(self):
        """
        `Encoding` object providing the base occupancy grid.

        See `trimesh.voxel.encoding` for implementations."""
        return self._data['encoding']

    @property
    def transform(self):
        return self._data['transform']

    @transform.setter
    def transform(self, transform):
        if transform is None:
            transform = IDENTITY_TRANSFORM
        elif isinstance(transform, np.ndarray):
            transform = Transformation.from_matrix(transform)
        elif not isinstance(transform, Transformation):
            raise ValueError(
                'transform must be a matrix or Transformation, got %s'
                % str(transform))
        self._data['transform'] = transform

    def apply_transform(self, matrix):
        transform = self.transform.apply_transform(matrix)
        return Voxel(self.encoding, transform)

    @caching.cache_decorator
    def stripped(self):
        """
        Get a `Voxel` instance with empty planes from each face removed.

        Returns:
            translated `Voxel` instance with a base matrix with
            leading/trailing planes of zeros removed.
        """
        encoding, padding = self.encoding.stripped
        # ops.strip_array(self.encoding.dense)
        transform = self.transform
        translation = (
            self.indices_to_points(padding[:, 0]) - transform.translation)
        return Voxel(encoding, transform.apply_translation(translation))

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
        return self.encoding.shape

    @caching.cache_decorator
    def filled_count(self):
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

    @caching.cache_decorator
    def marching_cubes(self):
        """
        A marching cubes Trimesh representation of the voxels.

        No effort was made to clean or smooth the result in any way;
        it is merely the result of applying the scikit-image
        measure.marching_cubes function to self.matrix.

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
        volume = self.filled_count * self.transform.volume
        return volume

    @caching.cache_decorator
    def points(self):
        """
        The center of each filled cell as a list of points.

        Returns
        ----------
        points: (self.filled, 3) float, list of points
        """
        return self.transform.transform_points(
            self.sparse_indices.astype(float))

    @property
    def sparse_indices(self):
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
        transform = self.transform
        if transform is not IDENTITY_TRANSFORM:
            mesh = mesh.apply_transform(transform.matrix)
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
        points = self.transform.inverse.transform_points(points)
        return np.round(points).astype(int)

    def indices_to_points(self, indices):
        return self.transform.transform_points(indices.astype(float))

    def show(self, *args, **kwargs):
        """
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        """
        return self.as_boxes(kwargs.pop('colors', None)).show(*args, **kwargs)

    def copy(self):
        return Voxel(self.encoding.copy(), self.transform.copy())

    def revoxelize(self, shape):
        """Create a new Voxel object without rotations or shearing."""
        from .. import util
        shape = tuple(shape)
        bounds = self.bounds.copy()
        extents = self.extents
        # the following is necessary if grid_linspace changes aren't accepted
        # x, y, z = (
        #     np.linspace(bounds[0, i], bounds[1, i], s)
        #     for i, s in enumerate(shape))
        # points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        points = util.grid_linspace(bounds, shape).reshape(shape + (3,))
        dense = self.is_filled(points)
        scale = extents / np.asanyarray(shape)
        translate = bounds[0]
        return Voxel(
            dense,
            transform=tr.scale_and_translate(scale, translate)
        )
