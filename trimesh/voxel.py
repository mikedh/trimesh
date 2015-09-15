import numpy as np
from .grouping   import unique_float
from .points     import plot_points
from collections import deque

class Voxel:
    def __init__(self, mesh, pitch): 
        self._run = mesh_to_run(mesh, pitch)
        self._raw = None
        
    @property
    def raw(self):
        '''
        Generate a raw 3D boolean array from the internal run-length encoded data
        '''
        if self._raw is  None:
            self._raw = run_to_raw(**self.run)
        return self._raw
         
    @property
    def run(self):
        return self._run
        
    @property
    def pitch(self):
        return self.run['pitch']

    @property
    def origin(self):
        return self.run['origin']

    def volume(self):
        volume = self.raw.sum() * (self.pitch**2)
        return volume
        
    def show(self):
        plot_raw(self.raw, **self.run)

def run_to_raw(shape, index_xy, index_z, **kwargs):
    raw = np.zeros(shape, dtype=np.bool)
    for xy, z in zip(index_xy, index_z):
        for z_start, z_end in np.reshape(z,(-1,2)):
            raw[xy[0], xy[1]][z_start:z_end] = True
    return raw

def mesh_to_run(mesh, pitch):
    '''
    Convert a mesh to a run-length encoded voxel grid. 

    This is done via ray tests which return intersection points,
    which is easily convertable to a raw 3D boolean voxel array
    '''
    bounds    = mesh.bounds / pitch
    bounds[0] = np.floor(bounds[0]) * pitch
    bounds[1] = np.ceil( bounds[1]) * pitch
    
    x_grid = np.arange(*bounds[:,0], step = pitch)
    y_grid = np.arange(*bounds[:,1], step = pitch)    
    grid   = np.dstack(np.meshgrid(x_grid, y_grid)).reshape((-1,2))

    ray_origins  = np.column_stack((grid, np.tile(bounds[0][2], len(grid))))
    ray_origins += [pitch*.5, pitch*.5, -pitch]
    ray_vectors  = np.tile([0.0,0.0,1.0], (len(grid),1))
    rays         = np.column_stack((ray_origins, 
                                    ray_vectors)).reshape((-1,2,3))

    hits        = mesh.ray.intersects_location(rays)    
    raw_shape   = np.ptp(bounds/pitch, axis=0).astype(int)
    grid_origin = bounds[0]
    grid_index  = ((grid/pitch) - (grid_origin[0:2]/pitch)).astype(int)

    run_z  = deque()
    run_xy = deque()

    for i, hit in enumerate(hits):
        if len(hit) == 0: continue

        z = hit[:,2]-grid_origin[2]
        # if a ray hits exactly on an edge, there will
        # be a duplicate entry in hit (for both triangles) 
        z = unique_float(z)

        index_z = np.round(z/pitch).astype(int)
        # if a hit is on edge, it will be returned twice
        # this np.unique call returns sorted, unique indicies
        index_z.sort()

        if np.mod(len(index_z), 2) != 0:
            # this is likely the on-vertex case
            index_z = index_z[[0,-1]]

        run_z.append(index_z)
        run_xy.append(grid_index[i])

    result = {'shape'    : raw_shape, 
              'index_xy' : np.array(run_xy), 
              'index_z'  : np.array(run_z), 
              'origin'   : grid_origin,
              'pitch'    : pitch}    
    return result        

def plot_raw(raw, pitch, origin, **kwargs):
    render = np.column_stack(np.nonzero(raw))*pitch + origin
    plot_points(render)
