"""
voxel.py
-----------

Convert meshes to a simple voxel data structure and back again.
"""
import numpy as np

from . import util
from . import remesh
from . import grouping

from .constants import log

from collections import deque


class Voxel(object):

    def __init__(self, *args, **kwargs):
        self._data = util.DataStore()
        self._cache = util.Cache(id_function=self._data.crc)

    @util.cache_decorator
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
        meshed = matrix_to_marching_cubes(matrix=self.matrix,
                                          pitch=self.pitch,
                                          origin=self.origin)
        return meshed

    @property
    def pitch(self):
        # stored as TrackedArray with a single element
        return self._data['pitch'][0]

    @pitch.setter
    def pitch(self, value):
        self._data['pitch'] = value

    @property
    def shape(self):
        """
        The shape of the matrix for the current voxel object.

        Returns
        ---------
        shape: (3,) int, what is the shape of the 3D matrix for these voxels
        """
        return self.matrix.shape

    @util.cache_decorator
    def filled_count(self):
        """
        Return the number of voxels that are occupied.

        Returns
        --------
        filled: int, number of voxels that are occupied
        """
        return int(self.matrix.sum())

    @util.cache_decorator
    def volume(self):
        """
        What is the volume of the filled cells in the current voxel object.

        Returns
        ---------
        volume: float, volume of filled cells
        """
        volume = self.filled_count * (self.pitch**3)
        return volume

    @util.cache_decorator
    def points(self):
        """
        The center of each filled cell as a list of points.

        Returns
        ----------
        points: (self.filled, 3) float, list of points
        """
        points = matrix_to_points(matrix=self.matrix,
                                  pitch=self.pitch,
                                  origin=self.origin)
        return points

    def point_to_index(self, point):
        """
        Convert a point to an index in the matrix array.

        Parameters
        ----------
        point: (3,) float, point in space

        Returns
        ---------
        index: (3,) int tuple, index in self.matrix
        """
        point = np.asanyarray(point)
        if point.shape != (3,):
            raise ValueError('to_index requires a single point')
        index = np.round((point - self.origin) / self.pitch).astype(int)
        index = tuple(index)
        return index

    def is_filled(self, point):
        """
        Query a point to see if the voxel cell it lies in is filled or not.

        Parameters
        ----------
        point: (3,) float, point in space

        Returns
        ---------
        is_filled: bool, is cell occupied or not
        """
        index = self.point_to_index(point)
        in_range = (np.array(index) < np.array(self.shape)).all()
        if in_range:
            is_filled = self.matrix[index]
        else:
            is_filled = False
        return is_filled


class VoxelMesh(Voxel):

    def __init__(self, mesh, pitch, max_iter=10, size_max=None):
        """
        A voxel representation of a mesh that will track changes to
        the mesh.

        At the moment the voxels are not filled in and only represent
        the surface.

        Parameters
        ----------
        mesh:      Trimesh object
        pitch:     float, how long should each edge of the voxel be
        size_max:  float, maximum size (in mb) of a data structure that
                          may be created before raising an exception
        """
        super(VoxelMesh, self).__init__()

        self._data['mesh'] = mesh
        self._data['pitch'] = pitch
        self._data['max_iter'] = max_iter

    @util.cache_decorator
    def matrix_surface(self):
        """
        The voxels on the surface of the mesh as a 3D matrix.

        Returns
        ---------
        matrix: self.shape np.bool, if a cell is True it is occupied
        """
        matrix = sparse_to_matrix(self.sparse_surface)
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
        # once filled voxels are implemented
        # if self._data['mesh'].is_watertight:
        #    return self.matrix_filled

        return self.matrix_surface

    @property
    def origin(self):
        populate = self.sparse_surface
        return self._cache['origin']

    @util.cache_decorator
    def sparse_surface(self):
        voxels, origin = voxelize_subdivide(mesh=self._data['mesh'],
                                            pitch=self._data['pitch'],
                                            max_iter=self._data['max_iter'][0])
        self._cache['origin'] = origin
        return voxels

    @util.cache_decorator
    def as_boxes(self):
        """
        A rough Trimesh representation of the voxels with a box for each filled voxel.

        Returns
        ---------
        mesh: Trimesh object representing the current voxel object.
        """
        centers = (self.sparse_surface *
                   self.pitch).astype(np.float64) + self.origin
        mesh = multibox(centers=centers, pitch=self.pitch)
        return mesh

    def show(self):
        """
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        """
        self.as_boxes.show()


def voxelize_subdivide(mesh, pitch, max_iter=10):
    """
    Voxelize a surface by subdividing a mesh until every edge is shorter
    than half the pitch of the voxel grid.

    Parameters
    -----------
    mesh:     Trimesh object
    pitch:    float, side length of a single voxel cube

    Returns
    -----------
    voxels_sparse:   (n,3) int, (m,n,p) indexes of filled cells
    origin_position: (3,) float, position of the voxel grid origin in space
    """
    max_edge = pitch / 2.0

    # get the same mesh sudivided so every edge is shorter than half our pitch
    v, f = remesh.subdivide_to_size(mesh.vertices,
                                    mesh.faces,
                                    max_edge=max_edge,
                                    max_iter=max_iter)

    # convert the vertices to their voxel grid position
    hit = (v / pitch)
    # get a conservative-ish wrapping
    hit = np.vstack((np.ceil(hit), np.floor(hit))).astype(int)

    # remove duplicates
    unique, inverse = grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = hit[unique]

    origin_index = occupied_index.min(axis=0)
    origin_position = origin_index * pitch

    voxels_sparse = (occupied_index - origin_index)

    return voxels_sparse, origin_position


def matrix_to_points(matrix, pitch, origin):
    """
    Convert an (n,m,p) matrix into a set of points for each voxel center.

    Parameters
    -----------
    matrix: (n,m,p) bool, voxel matrix
    pitch: float, what pitch was the voxel matrix computed with
    origin: (3,) float, what is the origin of the voxel matrix

    Returns
    ----------
    points: (q, 3) list of points
    """
    points = np.column_stack(np.nonzero(matrix)) * pitch + origin
    return points


def matrix_to_marching_cubes(matrix, pitch, origin):
    """
    Convert an (n,m,p) matrix into a mesh, using marching_cubes.

    Parameters
    -----------
    matrix: (n,m,p) bool, voxel matrix
    pitch: float, what pitch was the voxel matrix computed with
    origin: (3,) float, what is the origin of the voxel matrix

    Returns
    ----------
    mesh: Trimesh object, generated by meshing voxels using
                          the marching cubes algorithm in skimage
    """
    from skimage import measure
    from .base import Trimesh

    matrix = np.asanyarray(matrix, dtype=np.bool)

    rev_matrix = np.logical_not(matrix)  # Takes set about 0.
    # Add in padding so marching cubes can function properly with
    # voxels on edge of AABB
    pad_width = 1
    rev_matrix = np.pad(rev_matrix,
                        pad_width=(pad_width),
                        mode='constant',
                        constant_values=(1))

    # pick between old and new API
    if hasattr(measure, 'marching_cubes_lewiner'):
        func = measure.marching_cubes_lewiner
    else:
        func = measure.marching_cubes

    # Run marching cubes.
    meshed = func(volume=rev_matrix,
                  level=.5,  # it is a boolean voxel grid
                  spacing=(pitch,
                           pitch,
                           pitch))

    # allow results from either marching cubes function in skimage
    # binaries available for python 3.3 and 3.4 appear to use the classic
    # method
    if len(meshed) == 2:
        log.warning('using old marching cubes, may not be watertight!')
        vertices, faces = meshed
    elif len(meshed) == 4:
        vertices, faces, normals, vals = meshed

    # Return to the origin, add in the pad_width
    vertices = np.subtract(np.add(vertices, origin), pad_width)
    mesh = Trimesh(vertices=vertices, faces=faces)
    return mesh


def sparse_to_matrix(sparse):
    """
    Take a sparse (n,3) list of integer indexes of filled cells,
    turn it into a dense (m,o,p) matrix.

    Parameters
    -----------
    sparse: (n,3) int, index of filled cells

    Returns
    ------------
    dense: (m,o,p) bool, matrix of filled cells
    """

    sparse = np.asanyarray(sparse, dtype=np.int)
    if not util.is_shape(sparse, (-1, 3)):
        raise ValueError('sparse must be (n,3)!')

    shape = sparse.max(axis=0) + 1
    matrix = np.zeros(np.product(shape), dtype=np.bool)
    multiplier = np.array([np.product(shape[1:]), shape[2], 1])

    index = (sparse * multiplier).sum(axis=1)
    matrix[index] = True

    dense = matrix.reshape(shape)
    return dense


def multibox(centers, pitch):
    """
    Return a Trimesh object with a box at every center.

    Doesn't do anything nice or fancy.

    Parameters
    -----------
    centers: (n,3) float, center of boxes that are occupied
    pitch:   float, the edge length of a voxel

    Returns
    ---------
    rough: Trimesh object representing inputs
    """
    from . import primitives
    from .base import Trimesh

    b = primitives.Box(extents=[pitch, pitch, pitch])

    v = np.tile(centers, (1, len(b.vertices))).reshape((-1, 3))
    v += np.tile(b.vertices, (len(centers), 1))

    f = np.tile(b.faces, (len(centers), 1))
    f += np.tile(np.arange(len(centers)) * len(b.vertices),
                 (len(b.faces), 1)).T.reshape((-1, 1))

    rough = Trimesh(vertices=v, faces=f)

    return rough


def sparse_surface_to_filled(sparse_surface):
    """
    Take a sparse surface and fill in along Z.
    """
    pass
