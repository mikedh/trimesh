import numpy as np

from . import util
from . import grouping

from collections import deque

class Voxel:
    def __init__(self, mesh, pitch, size_max=1e5):
        '''
        Create a voxel representation of a mesh through ray tests
        which are done immediatly on instantiation.
        
        Arguments
        ----------
        mesh:  Trimesh object
        pitch: float, how long should each edge of the voxel be
        '''
        self._run   = mesh_to_run(mesh, pitch, size_max=size_max)     
        self._cache = util.Cache(id_function = self._id)
        
    def _id(self):
        '''
        Invalidate the cache if the current run object has been replaced. 
        Note that this is not monitoring self._run for changes through a DataStore
        '''
        return id(self._run)
                 
    @property
    def run(self):
        '''
        A dictionary holding data describing 'runs' of voxels. 
        
        This is a fairly natural and somewhat compact way of representing the voxels,
        as we generated the voxelization by doing ray tests along the Z axis.
        
        Note that this representation makes a lot of sense for watertight meshes but for 
        things that look more like random noise it makes basically no sense whatsoever.
        
        Returns
        -----------
        run: dictionary containing the following:
             'index_xy' : (n,2) int, containing XY position in voxel counts
             'index_z'  : (n,2) int, containing start and end of a voxel run, in Z
             'pitch'    : float, pitch voxels were computed at
             'origin'   : (3,) float, position in world space the voxel matrix origin is
             'shape'    : (3,) int, shape of voxel matrix
        '''
        return self._run
        
    @property
    def pitch(self):
        '''
        Rhe edge length of each voxel cube for the current object.
        
        Returns
        ----------
        pitch: float, length of an edge on a  single voxel cube
        '''
        return self.run['pitch']

    @property
    def origin(self):
        '''
        What is the origin of the current voxel object.
        
        Returns
        ---------
        origin: (3,) float, where the minimum corner for the current voxel
                 matrix originates. 
        '''
        return self.run['origin']

    @property
    def shape(self):
        '''
        The shape of the matrix for the current voxel object.
        
        Returns
        ---------
        shape: (3,) int, what is the shape of the 3D matrix for these voxels 
        '''
        return tuple(self.run['shape'])
        
    @util.cache_decorator
    def filled_count(self):
        '''
        Return the number of voxels that are occupied.
        
        Returns
        --------
        filled: int, number of voxels that are occupied
        '''
        if len(self.run['index_z']) > 0:
            filled = np.abs(np.diff(self.run['index_z'], axis=1)).sum()
        else:
            filled = 0
        return filled
        
    @util.cache_decorator
    def raw(self):
        '''
        The voxels represented as a 3D matrix.
        
        Returns
        ---------
        raw: self.shape np.bool, if a cell is True it is occupied
        '''
        return run_to_raw(**self.run)
        
    @util.cache_decorator
    def volume(self):
        '''
        What is the volume of the filled cells in the current voxel object.
        
        Returns
        ---------
        volume: float, volume of filled cells
        '''
        volume = self.filled_count * (self.pitch**3)
        return volume
        
    @util.cache_decorator
    def points(self):
        '''
        The center of each filled cell as a list of points.
        
        Returns
        ----------
        points: (self.filled, 3) float, list of points
        '''
        points = raw_to_points(raw    = self.raw, 
                               pitch  = self.pitch, 
                               origin = self.origin)
        return points
        
    @util.cache_decorator
    def mesh(self):
        '''
        A rough Trimesh representation of the voxels. 
        
        No effort was made to clean or smooth the result in any way; it is merely
        a set of columns for each Z run, although each column should essentially be 
        a separate body.

        Returns
        ---------
        mesh: Trimesh object representing the current voxel object.
        '''
        mesh = run_to_mesh(self.run)
        return mesh

    def point_to_index(self, point):
        '''
        Convert a point to an index in the raw array.

        Arguments
        ----------
        point: (3,) float, point in space

        Returns
        ---------
        index: (3,) int tuple, index in self.raw
        '''
        point = np.asanyarray(point)
        if point.shape != (3,):
            raise ValueError('to_index requires a single point')
        index = np.round((point - self.origin) / self.pitch).astype(int)
        index = tuple(index)
        return index
    
    def is_filled(self, point):
        '''
        Query a point to see if the voxel cell it lies in is filled or not.

        Arguments
        ----------
        point: (3,) float, point in space

        Returns
        ---------
        is_filled: bool, is cell occupied or not
        '''
        index = self.point_to_index(point)
        in_range = (np.array(index) < np.array(self.shape)).all()
        if in_range:
            is_filled = self.raw[index]
        else:
            is_filled = False
        return is_filled
        
    def show(self):
        '''
        Convert the current set of voxels into a trimesh for visualization
        and show that via its built- in preview method.
        '''
        self.mesh.show()
    
def mesh_to_run(mesh, pitch, size_max=1e5):
    '''
    Compute a list of occupied voxels from a Trimesh using ray tests.
    
    Arguments
    -----------
    mesh:  Trimesh object
    pitch: float, how long should a voxel edge be
    
    Returns
    -----------
    run: dictionary containing the following:
         'index_xy' : (n,2) int, containing XY position in voxel counts
         'index_z'  : (n,2) int, containing start and end of a voxel run, in Z
         'pitch'    : float, pitch voxels were computed at
         'origin'   : (3,) float, position in world space the voxel matrix origin is
         'shape'    : (3,) int, shape of voxel matrix

    '''
    bounds    = mesh.bounds / pitch
    bounds[0] = np.floor(bounds[0]) * pitch
    bounds[1] = np.ceil( bounds[1]) * pitch
    
    size = np.product(np.diff(bounds, axis=0))
    
    if size > size_max:
        raise ValueError('Voxels would be larger than max size!')
    
    
    x_grid = np.arange(*bounds[:,0], step = pitch)
    y_grid = np.arange(*bounds[:,1], step = pitch)    
    grid   = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1,2))
    
    ray_origins  = np.column_stack((grid, np.tile(bounds[0][2], len(grid))))
    ray_origins += [pitch*.5, pitch*.5, -pitch]
    ray_vectors  = np.tile([0.0,0.0,1.0], (len(grid),1))
    rays         = np.column_stack((ray_origins, 
                                    ray_vectors)).reshape((-1,2,3))
    
    hits        = mesh.ray.intersects_location(rays)    
    raw_shape   = np.ceil(np.ptp(bounds/pitch, axis=0)).astype(int)
    grid_origin = bounds[0]
    grid_index  = np.rint((grid/pitch) - (grid_origin[0:2]/pitch)).astype(int)
    
    run_z  = deque()
    run_xy = deque()

    for i, hit in enumerate(hits):
        if len(hit) == 0: 
            continue

        z = hit[:,2]-grid_origin[2]
        # if a ray hits exactly on an edge, there will
        # be a duplicate entry in hit (for both triangles) 
        z = grouping.unique_float(z)

        index_z = np.round(z/pitch).astype(int)
        # if a hit is on edge, it will be returned twice
        # this np.unique call returns sorted, unique indicies
        index_z.sort()

        if np.mod(len(index_z), 2) != 0:
            # this is likely the on-vertex case
            index_z = index_z[[0,-1]]

        # we tile multiple XY entries if there is more than one pair of intersections
        # by doing this we ensure that both run_z and run_xy are (n,2)
        run_z.extend(index_z.reshape((-1,2)))
        run_xy.extend(np.tile(grid_index[i], (int(len(index_z)/2), 1)))

    run = {'shape'    : raw_shape, 
           'index_xy' : np.array(run_xy), 
           'index_z'  : np.array(run_z), 
           'origin'   : grid_origin,
           'pitch'    : pitch}
    return run        

def run_to_raw(shape, index_xy, index_z, **kwargs):
    '''
    Convert a set of voxel runs to a 3D numpy array.
    
    Arguments
    ----------
    shape: (3,) int, shape of resulting matrix
    index_xy: (n,2) int, position (in counts) of voxels
    index_z:  (n,2) int, start and end of each voxel run in Z
    
    Returns
    ----------
    raw: (shape) bool numpy.ndarray
    '''
    raw = np.zeros(shape, dtype=np.bool)
    for xy, z in zip(index_xy, index_z):
        raw[xy[0], xy[1]][z[0]:z[1]] = True
    return raw
   
def raw_to_points(raw, pitch, origin):
    '''
    Convert an (n,m,p) raw matrix into a set of points for each voxel center.
    
    Arguments
    -----------
    raw: (n,m,p) bool, voxel matrix
    pitch: float, what pitch was the voxel matrix computed with
    origin: (3,) float, what is the origin of the voxel matrix
    
    Returns
    ----------
    points: (q, 3) list of points
    '''
    points = np.column_stack(np.nonzero(raw))*pitch + origin
    return points
    
def run_to_mesh(run):
    '''
    Convert information about voxel runs into a rough Trimesh for visualization.
    
    Arguments
    ------------
    run: dict with voxel run information
    
    Returns
    ------------
    rough: Trimesh object representing the voxels described
    '''
    from .creation import box
    from .base import Trimesh
    
    columns = len(run['index_xy'])
    
    z_middle = run['index_z'].mean(axis=1) * run['pitch']
    z_height = run['index_z'].ptp(axis=1)  * run['pitch']
    
    offsets = (run['index_xy'] * run['pitch'])
    offsets = np.column_stack((offsets, z_middle)) + run['origin']
    
    b = box()
    vertices = np.tile(b.vertices, (columns, 1, 1))
    vertices[:,:,2] *= z_height.reshape((-1,1))
    vertices[:,:,0:2] *= run['pitch']
    vertices += offsets.reshape((-1,1,3))
    vertices = vertices.reshape((-1,3))
    
    faces = np.tile(b.faces, (columns, 1, 1))
    face_offset = (np.tile(np.arange(columns)*len(b.vertices),
                           (3,1)).T).reshape((-1,1,3))
    faces += face_offset
    faces = faces.reshape((-1,3))
    
    rough = Trimesh(vertices = vertices,
                    faces    = faces,
                    process  = False)
    return rough
