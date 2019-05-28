import numpy as np

from ..constants import log_time
from .. import remesh
from .. import grouping
from .. import util
from . import ops


@log_time
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


def local_voxelize(mesh,
                   point,
                   pitch,
                   radius,
                   fill=True,
                   **kwargs):
    """
    Voxelize a mesh in the region of a cube around a point. When fill=True,
    uses proximity.contains to fill the resulting voxels so may be meaningless
    for non-watertight meshes. Useful to reduce memory cost for small values of
    pitch as opposed to global voxelization.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Source geometry
    point : (3, ) float
      Point in space to voxelize around
    pitch :  float
      Side length of a single voxel cube
    radius : int
      Number of voxel cubes to return in each direction.
    kwargs : parameters to pass to voxelize_subdivide

    Returns
    -----------
    voxels : (m, m, m) bool
      Array of local voxels where m=2*radius+1
    origin_position : (3,) float
      Position of the voxel grid origin in space
    """
    from scipy import ndimage

    # make sure point is correct type/shape
    point = np.asanyarray(point, dtype=np.float64).reshape(3)
    # this is a gotcha- radius sounds a lot like it should be in
    # float model space, not int voxel space so check
    if not isinstance(radius, int):
        raise ValueError('radius needs to be an integer number of cubes!')

    # Bounds of region
    bounds = np.concatenate((point - (radius + 0.5) * pitch,
                             point + (radius + 0.5) * pitch))

    # faces that intersect axis aligned bounding box
    faces = list(mesh.triangles_tree.intersection(bounds))

    # didn't hit anything so exit
    if len(faces) == 0:
        return np.array([], dtype=np.bool), np.zeros(3)

    local = mesh.submesh([[f] for f in faces], append=True)

    # Translate mesh so point is at 0,0,0
    local.apply_translation(-point)

    sparse, origin = voxelize_subdivide(local, pitch, **kwargs)
    matrix = ops.sparse_to_matrix(sparse)

    # Find voxel index for point
    center = np.round(-origin / pitch).astype(np.int64)

    # pad matrix if necessary
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

        where = np.where(contains)[0] + 1
        # use in1d vs isin for older numpy versions
        internal = np.in1d(regions.flatten(), where).reshape(regions.shape)

        voxels = np.logical_or(voxels, internal)

    return voxels, local_origin


@log_time
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
