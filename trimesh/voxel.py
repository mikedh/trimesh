"""
voxel.py
-----------

Convert meshes to a simple voxel data structure and back again.
"""
import numpy as np

from . import util
from . import remesh
from . import caching
from . import grouping

from .constants import log, _log_time


class Voxel(object):

    def __init__(self, *args, **kwargs):
        self._data = caching.DataStore()
        self._cache = caching.Cache(id_function=self._data.crc)

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
        shape: (3,) int, what is the shape of the 3D matrix
                         for these voxels
        """
        return self.matrix.shape

    @caching.cache_decorator
    def filled_count(self):
        """
        Return the number of voxels that are occupied.

        Returns
        --------
        filled: int, number of voxels that are occupied
        """
        return int(self.matrix.sum())

    @caching.cache_decorator
    def volume(self):
        """
        What is the volume of the filled cells in the current voxel object.

        Returns
        ---------
        volume: float, volume of filled cells
        """
        volume = self.filled_count * (self.pitch**3)
        return volume

    @caching.cache_decorator
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
        index = np.round((point - self.origin) /
                         self.pitch).astype(int)
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

    def __init__(self,
                 mesh,
                 pitch,
                 max_iter=10,
                 size_max=None,
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
        size_max:  float, maximum size (in mb) of a data structure that
                          may be created before raising an exception
        """
        super(VoxelMesh, self).__init__()

        self._method = method
        self._data['mesh'] = mesh
        self._data['pitch'] = pitch
        self._data['max_iter'] = max_iter

    @caching.cache_decorator
    def matrix_surface(self):
        """
        The voxels on the surface of the mesh as a 3D matrix.

        Returns
        ---------
        matrix: self.shape np.bool, if a cell is True it is occupied
        """
        matrix = sparse_to_matrix(self.sparse_surface)
        return matrix

    @caching.cache_decorator
    def matrix_solid(self):
        """
        The voxels in a mesh as a 3D matrix.

        Returns
        ---------
        matrix: self.shape np.bool, if a cell is True it is occupied
        """
        matrix = sparse_to_matrix(self.sparse_solid)
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
        populate = self.sparse_surface
        return self._cache['origin']

    @caching.cache_decorator
    def sparse_surface(self):
        """
        Filled cells on the surface of the mesh.

        Returns
        ----------------
        voxels: (n, 3) int, filled cells on mesh surface
        """
        if self._method == 'ray':
            func = voxelize_ray
        elif self._method == 'subdivide':
            func = voxelize_subdivide
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
        filled = fill_voxelization(self.sparse_surface)
        return filled

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
        centers = (filled * self.pitch).astype(np.float64)
        centers += self.origin - (self.pitch / 2.0)
        mesh = multibox(centers=centers, pitch=self.pitch)
        return mesh

    def show(self, solid=False):
        """
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        """
        self.as_boxes(solid=solid).show()


@_log_time
def voxelize_subdivide(mesh,
                       pitch,
                       max_iter=10,
                       edge_factor=2.0):
    """
    Voxelize a surface by subdividing a mesh until every edge is
    shorter than: (pitch / edge_factor)

    Parameters
    -----------
    mesh:        Trimesh object
    pitch:       float, side length of a single voxel cube
    max_iter:    int, cap maximum subdivisions or None for no limit.
    edge_factor: float,

    Returns
    -----------
    voxels_sparse:   (n,3) int, (m,n,p) indexes of filled cells
    origin_position: (3,) float, position of the voxel
                                 grid origin in space
    """
    max_edge = pitch / edge_factor

    if max_iter is None:
        longest_edge = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] -
                                      mesh.vertices[mesh.edges[:, 1]],
                                      axis=1).max()
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)

    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    v, f = remesh.subdivide_to_size(mesh.vertices,
                                    mesh.faces,
                                    max_edge=max_edge,
                                    max_iter=max_iter)

    # convert the vertices to their voxel grid position
    hit = v / pitch

    # Provided edge_factor > 1 and max_iter is large enough, this is
    # sufficient to preserve 6-connectivity at the level of voxels.
    hit = np.round(hit).astype(int)

    # remove duplicates
    unique, inverse = grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = hit[unique]

    origin_index = occupied_index.min(axis=0)
    origin_position = origin_index * pitch

    voxels_sparse = (occupied_index - origin_index)

    return voxels_sparse, origin_position


def local_voxelize(mesh, point, pitch, radius, fill=True, **kwargs):
    """
    Voxelize a mesh in the region of a cube around a point. When fill=True,
    uses proximity.contains to fill the resulting voxels so may be meaningless
    for non-watertight meshes. Useful to reduce memory cost for small values of
    pitch as opposed to global voxelization.

    Parameters
    -----------
    mesh:   Trimesh object
    point:  (3, ) float, point in space
    pitch:  float, side length of a single voxel cube
    radius: int, number of voxel cubes to return in each direction.
    kwargs: parameters to pass to voxelize_subdivide

    Returns
    -----------
    voxels:          (m, m, m) bool, matrix of local voxels where m=2*radius+1
    origin_position: (3,) float, position of the voxel grid origin in space
    """
    from scipy import ndimage

    point = np.asanyarray(point, dtype=np.float64)

    # Bounds of region
    bounds = np.concatenate((point - (radius + 0.5) * pitch,
                             point + (radius + 0.5) * pitch))

    # faces that intersect axis aligned bounding box
    faces = list(mesh.triangles_tree.intersection(bounds))
    local = mesh.submesh([[f] for f in faces], append=True)

    # Translate mesh so point is at 0,0,0
    local.apply_translation(-point)

    sparse, origin = voxelize_subdivide(local, pitch, **kwargs)
    matrix = sparse_to_matrix(sparse)

    # Find voxel index for point
    center = np.round(-origin / pitch).astype(int)

    # Pad matrix if neccesary
    prepad = np.maximum(radius - center, 0)
    postpad = np.maximum(center + radius + 1 - matrix.shape, 0)
    matrix = np.pad(matrix, np.stack((prepad, postpad), axis=-1),
                    mode='constant')
    center += prepad

    # Extract voxels within the bounding box
    voxels = matrix[center[0] - radius:center[0] + radius + 1,
                    center[1] - radius:center[1] + radius + 1,
                    center[2] - radius:center[2] + radius + 1]
    local_origin = point - radius * pitch  # origin of local voxels

    # Fill internal regions
    if fill:
        regions, n = ndimage.measurements.label(~voxels)
        distance = ndimage.morphology.distance_transform_cdt(~voxels)
        representatives = [np.unravel_index((distance * (regions == i)).argmax(),
                                            distance.shape) for i in range(1, n + 1)]
        contains = mesh.contains(
            np.asarray(representatives) *
            pitch +
            local_origin)
        internal = np.isin(regions, np.where(contains)[0] + 1)
        voxels = np.logical_or(voxels, internal)

    return voxels, local_origin


@_log_time
def voxelize_ray(mesh,
                 pitch,
                 per_cell=[2, 2],
                 **kwargs):
    """
    Voxelize a mesh using ray queries.

    Parameters
    -------------
    mesh     : Trimesh object
                 Mesh to be voxelized
    pitch    : float
                 Length of voxel cube
    per_cell : (2,) int
                 How many ray queries to make per cell

    Returns
    -------------
    voxels : (n, 3) int
                 Voxel positions
    origin : (3, ) int
                 Origin of voxels
    """
    # how many rays per cell
    per_cell = np.array(per_cell).astype(np.int).reshape(2)
    # edge length of cube voxels
    pitch = float(pitch)

    # create the ray origins in a grid
    bounds = mesh.bounds[:, :2].copy()
    # offset start so we get the requested number per cell
    bounds[0] += pitch / (1.0 + per_cell)
    # offset end so arange doesn't short us
    bounds[1] += pitch
    # on X we are doing multiple rays per voxel step
    step = pitch / per_cell
    # 2D grid
    ray_ori = util.grid_arange(bounds, step=step)
    # a Z position below the mesh
    z = np.ones(len(ray_ori)) * (mesh.bounds[0][2] - pitch)
    ray_ori = np.column_stack((ray_ori, z))
    # all rays are along positive Z
    ray_dir = np.ones_like(ray_ori) * [0, 0, 1]

    # if you have pyembree this should be decently fast
    hits = mesh.ray.intersects_location(ray_ori, ray_dir)[0]

    # just convert hit locations to integer positions
    voxels = np.round(hits / pitch).astype(np.int64)

    # offset voxels by min, so matrix isn't huge
    origin = voxels.min(axis=0)
    voxels -= origin

    return voxels, origin


def fill_voxelization(occupied):
    """
    Given a sparse surface voxelization, fill in between columns.

    Parameters
    --------------
    occupied: (n, 3) int, location of filled cells

    Returns
    --------------
    filled: (m, 3) int, location of filled cells
    """
    # validate inputs
    occupied = np.asanyarray(occupied, dtype=np.int64)
    if not util.is_shape(occupied, (-1, 3)):
        raise ValueError('incorrect shape')

    # create grid and mark inner voxels
    max_value = occupied.max() + 3

    grid = np.zeros((max_value,
                     max_value,
                     max_value),
                    dtype=np.int64)
    voxels_sparse = np.add(occupied, 1)

    grid.__setitem__(tuple(voxels_sparse.T), 1)

    for i in range(max_value):
        check_dir2 = False
        for j in range(0, max_value - 1):
            idx = []
            # find transitions first
            # transition positions are from 0 to 1 and from 1 to 0
            eq = np.equal(grid[i, j, :-1], grid[i, j, 1:])
            idx = np.where(np.logical_not(eq))[0] + 1
            c = len(idx)
            check_dir2 = (c % 4) > 0 and c > 4
            if c < 4:
                continue
            for s in range(0, c - c % 4, 4):
                grid[i, j, idx[s]:idx[s + 3]] = 1
        if not check_dir2:
            continue

        # check another direction for robustness
        for k in range(0, max_value - 1):
            idx = []
            # find transitions first
            eq = np.equal(grid[i, :-1, k], grid[i, 1:, k])
            idx = np.where(np.logical_not(eq))[0] + 1
            c = len(idx)
            if c < 4:
                continue
            for s in range(0, c - c % 4, 4):
                grid[i, idx[s]:idx[s + 3], k] = 1

    # generate new voxels
    idx = np.where(grid == 1)
    filled = np.array([[idx[0][i] - 1,
                        idx[1][i] - 1,
                        idx[2][i] - 1]
                       for i in range(len(idx[0]))])

    return filled


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
        normals = None
    elif len(meshed) == 4:
        vertices, faces, normals, vals = meshed

    # Return to the origin, add in the pad_width
    vertices = np.subtract(np.add(vertices, origin), pad_width * pitch)
    # create the mesh
    mesh = Trimesh(vertices=vertices,
                   faces=faces,
                   vertex_normals=normals)
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


def boolean_sparse(a, b, operation=np.logical_and):
    """
    Find common rows between two arrays very quickly
    using 3D boolean sparse matrices.

    Parameters
    -----------
    a: (n, d)  int, coordinates in space
    b: (m, d)  int, coordinates in space
    operation: numpy operation function, ie:
                  np.logical_and
                  np.logical_or

    Returns
    -----------
    coords: (q, d) int, coordinates in space
    """
    # 3D sparse arrays, using wrapped scipy.sparse
    # pip install sparse
    import sparse

    # find the bounding box of both arrays
    extrema = np.array([a.min(axis=0),
                        a.max(axis=0),
                        b.min(axis=0),
                        b.max(axis=0)])
    origin = extrema.min(axis=0) - 1
    size = tuple(extrema.ptp(axis=0) + 2)

    # put nearby voxel arrays into same shape sparse array
    sp_a = sparse.COO((a - origin).T,
                      data=np.ones(len(a), dtype=np.bool),
                      shape=size)
    sp_b = sparse.COO((b - origin).T,
                      data=np.ones(len(b), dtype=np.bool),
                      shape=size)

    # apply the logical operation
    # get a sparse matrix out
    applied = operation(sp_a, sp_b)
    # reconstruct the original coordinates
    coords = np.column_stack(applied.coords) + origin

    return coords
