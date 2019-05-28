"""
voxel.py
-----------

Convert meshes to a simple voxel data structure and back again.
"""
import abc
import numpy as np

from .. import caching
from . import ops
from ..constants import log
from ..parent import Geometry
from .. import transformations


def _tuple(axes):
    if isinstance(axes, np.ndarray):
        return tuple(a.item() for a in axes)
    else:
        return tuple(axes)


SWITCH_YZ = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)

SWITCH_YZ.flags.writeable = False


class VoxelBase(Geometry):

    def __init__(self, *args, **kwargs):
        self._data = caching.DataStore()
        self._cache = caching.Cache(id_function=self._data.crc)

    def apply_transform(self, matrix):
        return TransformedVoxel(self, matrix)

    @caching.cache_decorator
    def bounds(self):
        points = self.points
        bounds = np.array([points.min(axis=0),
                           points.max(axis=0)])
        bounds.flags.writeable = False
        return bounds

    @caching.cache_decorator
    def extents(self):
        extents = self.bounds.ptp(axis=0)
        extents.flags.writeable = False
        return extents

    @abc.abstractproperty
    def is_empty(self):
        pass

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
        meshed = ops.matrix_to_marching_cubes(matrix=self.matrix)
        return meshed

    @property
    def pitch(self):
        # would rather just expose `transform`
        return 1.0

    @property
    def origin(self):
        # would rather just expose `transform`
        return np.zeros((3,))

    @property
    def voxel_volume(self):
        return 1.0

    @property
    def transform(self):
        # return np.eye(4)
        return None

    @abc.abstractproperty
    def shape(self):
        """
        The shape of the matrix for the current voxel object.

        Returns
        ---------
        shape: (3,) int, what is the shape of the 3D matrix
                         for these voxels
        """
        pass

    @abc.abstractproperty
    def filled_count(self):
        """
        Return the number of voxels that are occupied.

        Returns
        --------
        filled: int, number of voxels that are occupied
        """
        pass
        return int(self.matrix.sum())

    @caching.cache_decorator
    def volume(self):
        """
        What is the volume of the filled cells in the current voxel object.

        Returns
        ---------
        volume: float, volume of filled cells
        """
        volume = self.filled_count * (self.voxel_volume)
        return volume

    @abc.abstractproperty
    def points(self):
        """
        The center of each filled cell as a list of points.

        Returns
        ----------
        points: (self.filled, 3) float, list of points
        """
        pass

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def to_dense(self):
        pass

    def as_boxes(self):
        return self.to_dense().as_boxes()

    def point_to_index(self, point):
        """
        Convert points to indices in the matrix array.

        Parameters
        ----------
        point: (..., 3) float, point in space

        Returns
        ---------
        index: (..., 3) int array of indices into self.matrix
        """
        return np.round(point).astype(int)

    def show(self, *args, **kwargs):
        """
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        """
        return self.as_boxes().show(*args, **kwargs)


class Voxel(VoxelBase):
    """
    Voxel representation with dense matrix.

    All voxels are referenced by the center of their cell box, for example,
    the ``origin`` exactly matches the center of the first voxel.

    Parameters
    ----------
    matrix: (X, Y, Z)
        Matrix that is interpreted as boolean to represent filled voxels.
    pitch: float
        Scale of each voxel.
    origin: (3,) float
        The center of the first voxel.
    """

    def __init__(self, matrix):
        super(Voxel, self).__init__()
        if matrix is not None:
            self._data['matrix'] = matrix

    @caching.cache_decorator
    def strip(self):
        """
        Strip empty planes from each face.

        Returns:
            translated Voxel instance with a base matrix with leading/trailing
            planes of zeros removed.
        """
        matrix, padding = ops.strip_array(self.matrix)
        return Voxel(matrix).apply_translation(padding[:, 0].astype(float))

    def to_dense(self):
        return self

    @caching.cache_decorator
    def is_empty(self):
        return not np.any(self.matrix)

    def copy(self):
        return Voxel(self.matrix)

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def filled_count(self):
        return np.count_nonzero(self.matrix)

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
        out_shape = point.shape[:-1]
        point = point.reshape(-1, 3)
        index = self.point_to_index(point)
        in_range = np.logical_and(
            np.all(index < np.array(self.shape), axis=-1),
            np.all(index >= 0, axis=-1))

        is_filled = np.zeros_like(in_range)
        # get flat indices of those points in range
        flat_index = np.ravel_multi_index(index[in_range].T, self.shape)
        is_filled[in_range] = self.matrix.flat[flat_index]
        return is_filled.reshape(out_shape)

    @property
    def matrix(self):
        return self._data['matrix']

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
        matrix = self._data['matrix']
        centers = ops.matrix_to_points(
            matrix=matrix,
            pitch=self._data['pitch'],
            origin=self._data['origin'])

        if colors is not None:
            colors = np.asanyarray(colors)
            if (colors.ndim == 4 and
                colors.shape[:3] == matrix.shape and
                    colors.shape[3] in [3, 4]):
                colors = colors[matrix > 0]
            elif not (colors.shape == (3,) or colors.shape == (4,)):
                log.warning('colors incorrect shape!')
                colors = None

        mesh = ops.multibox(centers=centers,
                            pitch=self.pitch,
                            colors=colors)
        return mesh

    @property
    def points(self):
        """
        The center of each filled cell as a list of points.

        Returns
        ----------
        points: (self.filled, 3) float, list of points
        """
        indices = np.column_stack(np.nonzero(self.matrix))
        return indices.astype(float)


class VoxelMesh(Voxel):

    def __init__(self,
                 mesh,
                 pitch,
                 max_iter=10,
                 method='subdivide'):
        """
        A voxel representation of a mesh that will track changes to
        the mesh.

        At the moment the voxels are not filled in and only represent
        the surface.

        Parameters
        ----------
        mesh:      Trimesh object
        pitch:     float, how long should each edge of the voxel be
        """
        super(VoxelMesh, self).__init__(matrix=None)

        self._method = method
        self._data['mesh'] = mesh
        self._data['pitch'] = pitch
        self._data['max_iter'] = max_iter

    @property
    def pitch(self):
        return float(self._data['pitch'])

    @caching.cache_decorator
    def is_empty(self):
        return self.mesh.is_empty

    def copy(self):
        return VoxelMesh(
            mesh=self._data['mesh'],
            pitch=self._data['pitch'],
            max_iter=self._data['max_iter'],
            method=self._method)

    @caching.cache_decorator
    def matrix_surface(self):
        """
        The voxels on the surface of the mesh as a 3D matrix.

        Returns
        ---------
        matrix: self.shape np.bool, if a cell is True it is occupied
        """
        matrix = ops.sparse_to_matrix(self.sparse_surface)
        return matrix

    @caching.cache_decorator
    def matrix_solid(self):
        """
        The voxels in a mesh as a 3D matrix.

        Returns
        ---------
        matrix: self.shape np.bool, if a cell is True it is occupied
        """
        matrix = ops.sparse_to_matrix(self.sparse_solid)
        return matrix

    @property
    def matrix(self):
        """
        A matrix representation of the surface voxels.

        In the future this is planned to return a filled voxel matrix
        if the source mesh is watertight, and a surface voxelization
        otherwise.

        Returns
        ---------
        matrix: self.shape np.bool, cell occupancy
        """
        if self._data['mesh'].is_watertight:
            return self.matrix_solid
        return self.matrix_surface

    @property
    def origin(self):
        """
        The origin of the voxel array.

        Returns
        ------------
        origin: (3,) float, point in space
        """
        populate = self.sparse_surface  # NOQA
        return self._cache['origin']

    @caching.cache_decorator
    def sparse_surface(self):
        """
        Filled cells on the surface of the mesh.

        Returns
        ----------------
        voxels: (n, 3) int, filled cells on mesh surface
        """
        from . import creation
        if self._method == 'ray':
            func = creation.voxelize_ray
        elif self._method == 'subdivide':
            func = creation.voxelize_subdivide
        else:
            raise ValueError('voxelization method incorrect')

        voxels, origin = func(
            mesh=self._data['mesh'],
            pitch=self._data['pitch'],
            max_iter=self._data['max_iter'][0])
        self._cache['origin'] = origin

        return voxels

    @caching.cache_decorator
    def sparse_solid(self):
        """
        Filled cells inside and on the surface of mesh

        Returns
        ----------------
        filled: (n, 3) int, filled cells in or on mesh.
        """
        filled = ops.fill_voxelization(self.sparse_surface)
        return filled + 0.5

    def as_boxes(self, solid=False):
        """
        A rough Trimesh representation of the voxels with a box
        for each filled voxel.

        Parameters
        -----------
        solid: bool, if True return boxes for sparse_solid

        Returns
        ---------
        mesh: Trimesh object made up of one box per filled cell.
        """
        if solid:
            filled = self.sparse_solid
        else:
            filled = self.sparse_surface
        # center points of voxels
        centers = filled.astype(float) * self.pitch + self.origin
        mesh = ops.multibox(centers=centers, pitch=self.pitch)
        return mesh

    def show(self, *args, **kwargs):
        """
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        """
        self.as_boxes(solid=kwargs.pop('solid', None)).show(*args, **kwargs)

    @property
    def voxel_volume(self):
        return self.pitch**3


class VoxelRle(VoxelBase):
    """Run-length-encoded voxel."""
    def __init__(self, encoding, shape):
        """
        Args:
            encoding: `trimesh.rle.RunLengthEncoding` instance denoting 1D
                representation in `x, y, z` ordering.
            pitch: length of each voxel side length
            origin: length 3 float
            shape: shape of voxel
        """
        super(VoxelRle, self).__init__()
        self._encoding = encoding
        self._shape = tuple(shape)

    @caching.cache_decorator
    def is_empty(self):
        return self._encoding.sum() == 0

    def copy(self):
        return VoxelRle(encoding=self._encoding, shape=self._shape)

    @property
    def filled_count(self):
        return self.encoding.sum()

    @property
    def encoding(self):
        return self._encoding

    @property
    def shape(self):
        return self._shape

    def is_filled(self, point):
        point = np.asanyarray(point)
        out_shape = point.shape[:-1]
        point = point.reshape(-1, 3)
        index = np.asanyarray(self.point_to_index(point))
        in_range = np.logical_and(
            np.all(index < np.array(self.shape), axis=-1),
            np.all(index >= 0, axis=-1))
        is_filled = np.zeros_like(in_range)
        if np.any(in_range):
            flat_index = np.ravel_multi_index(index[in_range].T, self.shape)
            is_filled[in_range] = self.encoding.gather(flat_index)
        return is_filled.reshape(out_shape)

    @caching.cache_decorator
    def points(self):
        indices_1d = self.encoding.sparse_indices()
        indices_3d = np.stack(
            np.unravel_index(indices_1d, self.shape), axis=-1)
        return indices_3d.astype(float)

    @caching.cache_decorator
    def matrix(self):
        return self.encoding.to_dense().reshape(self.shape)

    def to_dense(self):
        """Convert to a Voxel representation based on a dense matrix."""
        return Voxel(matrix=self.matrix)

    @staticmethod
    def from_binvox_data(
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
            `VoxelBase` instance: `VoxelRle` or `VoxelTranspose` instance if
            `axis_order` isn't quivalent to 'xyz' or (0, 1, 2).
        """
        # shape must be uniform else scale is ambiguous
        from . import runlength as rl
        if not (shape[0] == shape[1] == shape[2]):
            raise ValueError(
                'trimesh only supports uniform scaling, so required binvox '
                'with uniform shapes')
        if isinstance(rle_data, rl.RunLengthEncoding):
            encoding = rle_data
        else:
            encoding = rl.RunLengthEncoding(rle_data)
        if scale == 1.0 and (translate is None or all(
                t == 0 for t in translate)):
            transform = None
        else:
            transform = transformations.scale_matrix(scale)
            transform[:3, 3] = translate
        if axis_order == 'xzy':
            if transform is None:
                transform = SWITCH_YZ
            else:
                transform = np.matmul(SWITCH_YZ, transform)
        elif axis_order is None or axis_order == 'xyz':
            pass
        else:
            raise ValueError(
                "Invalid axis_order '%s': must be None, 'xyz' or 'xzy'")

        voxel = VoxelRle(encoding, shape)
        if transform is not None:
            voxel = voxel.apply_transform(transform)

        return voxel


class TransformedVoxel(VoxelBase):
    def __init__(self, base, transform):
        super().__init__()
        self._base = base
        if transform.shape != (4, 4):
            raise ValueError(
                'transform must be 4x4 matrix, got shape %s '
                % str(transform.shape))
        self._data['transform'] = transform
        assert(hasattr(transform, 'shape'))

    @caching.cache_decorator
    def is_empty(self):
        return self.base.is_empty

    def copy(self):
        return TransformedVoxel(self.base.copy(), self.transform.copy())

    def apply_transform(self, matrix):
        if not hasattr(matrix, 'shape'):
            print(matrix)
        return TransformedVoxel(
            self.base, np.matmul(matrix, self.transform))

    @property
    def pitch(self):
        return np.power(self.voxel_volume, 1./3)

    @property
    def origin(self):
        return self.transform[:3, 3]

    @property
    def shape(self):
        return self.base.shape

    @property
    def base(self):
        return self._base

    @caching.cache_decorator
    def voxel_volume(self):
        return np.linalg.det(self.transform[:3, :3])*self.base.voxel_volume

    @property
    def transform(self):
        return self._data['transform']

    @caching.cache_decorator
    def marching_cubes(self):
        return self.base.marching_cubes.apply_transform(self.transform)

    @property
    def filled_count(self):
        return self.base.filled_count

    @caching.cache_decorator
    def points(self):
        return self._transform_points(self.base.points)

    def _transform_points(self, points, inverse=False):
        if len(points.shape) == 1:
            return np.squeeze(
                self._transform_points_2d(
                    np.expand_dims(points, axis=0), inverse=inverse),
                axis=0)
        else:
            return self._transform_points_2d(points, inverse=inverse)

    def _transform_points_2d(self, points, inverse=False):
        transform = self.inverse_transform if inverse else self.transform
        return transformations.transform_points(points, transform)

    @caching.cache_decorator
    def inverse_transform(self):
        return transformations.inverse_matrix(self.transform)

    def point_to_index(self, point):
        return self.base.point_to_index(
            self._transform_points(point, inverse=True))

    def to_dense(self):
        return self.base.to_dense().apply_transform(self.transform)

    def _is_filled(self, point):
        return self.base.is_filled(
            self._transform_points(point, inverse=True))

    def is_filled(self, point):
        shape = point.shape
        return self._is_filled(point.reshape((-1, 3))).reshape(shape[:-1])
