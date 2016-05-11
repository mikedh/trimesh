import numpy as np 

from shapely.geometry import Point, Polygon, LineString

from collections import deque

from ..constants import log, time_function
from ..constants import tol_path as tol

from .polygons  import polygons_obb, transform_polygon
from ..util     import transformation_2D

class RectangleBin:
    '''
    2D BSP tree node. 
    http://www.blackpawn.com/texts/lightmaps/
    '''
    def __init__(self, bounds=None, size=None):
        self.child  = [None] * 2

        #bounds: (minx, miny, maxx, maxy)
        self.bounds   = bounds
        self.occupied = False
        
        if size != None:  
            self.bounds = np.append([0,0], size)
        
    def insert(self, rectangle_size):
        for child in self.child:
            if child != None: 
                attempt = child.insert(rectangle_size)
                if attempt: return attempt
                
        if self.occupied:
            return None

        # compare the bin size to the insertion candidate size
        size_test = bounds_to_size(self.bounds) - rectangle_size

        # this means the inserted rectangle is too big for the cell
        if np.any(size_test < -tol.zero):
            return None
        
        # since the cell is big enough for the current rectangle, either it
        # is going to be inserted here, or the cell is going to be split
        # either way, the cell is now occupied. 
        self.occupied = True    
        
        # this means the inserted rectangle fits perfectly
        # since we already checked to see if it was negative, no abs is needed
        if np.all(size_test < tol.zero): 
            return self.bounds[0:2]
        
        # since the rectangle fits but the empty space is too big, 
        # we need to create some children to insert into
        # first, we decide which way to split
        vertical      = size_test[0] > size_test[1]
        length        = rectangle_size[not vertical]
        child_bounds  = self.split(length, vertical)

        self.child[0] = RectangleBin(bounds=child_bounds[0])
        self.child[1] = RectangleBin(bounds=child_bounds[1])
        
        return self.child[0].insert(rectangle_size)
   
    def split(self, length=None, vertical=True):
        '''
        returns two bounding boxes representing the current
        bounds split into two smaller boxes
        '''
        #also know as [minx, miny, maxx, maxy]
        [left, bottom, right, top] = self.bounds
        if vertical:
            box = [[left,        bottom, left+length, top],
                   [left+length, bottom, right,       top]]
        else: 
            box = [[left, bottom,          right, bottom + length],
                   [left, bottom + length, right, top]]
        return box

def bounds_to_size(bounds):
    return np.diff(np.reshape(bounds, (2,2)), axis=0)[0]    

def pack_rectangles(rectangles, sheet_size, shuffle=False):
    '''
    Pack smaller rectangles onto a larger rectangle, using a binary space partition tree.

    Parameters
    ----------
    rectangles: (n,2) array of (width, height) pairs representing the smaller rectangles to be packed.
    sheet_size: (2) array of (width, height) pair representing the sheet size the smaller rectangles will be packed onto.
    shuffle: boolean, whether or not to shuffle the insert order of the smaller rectangles, as the final packing density depends on the order of which rectangles are inserted onto the larger sheet. 
    density_escape: float, at what density should the loop exit early and return. A value of zero will return immediately after a single rectangle is placed, and a value of >= 1.0 will always traverse the entire list of smaller rectangles.
    '''
    offset    = np.zeros((len(rectangles), 2))
    inserted  = np.zeros( len(rectangles), dtype=np.bool)
    box_order = np.argsort(np.sum(rectangles**2, axis=1))[::-1]
    area      = 0.0
    density   = 0.0
    
    if shuffle: 
        shuffle_len = int(np.random.random()* len(rectangles)) - 1  
        box_order[0:shuffle_len] = np.random.permutation(box_order[0:shuffle_len])

    sheet = RectangleBin(size=sheet_size)
    for index in box_order: 
        insert_location = sheet.insert(rectangles[index])
        if insert_location != None:
            area           += np.prod(rectangles[index])
            offset[index]  += insert_location
            inserted[index] = True
    
    consumed_box = np.max((offset + rectangles)[inserted], axis=0)
    density      = area / np.product(consumed_box) 
            
    return density, offset[inserted], inserted, consumed_box
  
def pack_paths(paths, show=False):
    paths_full = deque()
    for path in paths:
        if 'quantity' in path.metadata:
            paths_full.extend([path.copy() for i in range(path.metadata['quantity'])])
        else:
            paths_full.append(path.copy())

    polygons   = [i.polygons_closed[i.root[0]] for i in paths_full]
    inserted, transforms = multipack(np.array(polygons))
    for path, transform in zip(paths_full, transforms):
        path.transform(transform)
        if show: path.plot_discrete(show=False)
    if show: plt.show()
    return paths_full

def multipack(polygons, 
              sheet_size     = None,
              iterations     = 50,
              density_escape = .985,
              buffer_dist    = 0.09,
              plot           = False,
              return_all     = False):
    '''
    Run multiple iterations of rectangle packing, by randomly permutating the insertion order

    If sheet size isn't specified, it creates a large sheet that can fit all of the polygons
    '''
    transforms_obb, rectangles = polygons_obb(polygons)
    rectangles                += 2.0*buffer_dist
    polygon_area               = np.array([p.area for p in polygons])


    tic             = time_function()  
    overall_density = 0
    
    if sheet_size==None:
        max_dim    = np.max(rectangles, axis=0)
        sum_dim    = np.sum(rectangles, axis=0)
        sheet_size = [sum_dim[0], max_dim[1]*2]

    log.info('Packing %d polygons', len(polygons))
    for i in range(iterations):
        density, offset, inserted, sheet = pack_rectangles(rectangles, 
                                                           sheet_size = sheet_size, 
                                                           shuffle    = (i != 0))
        if density > overall_density:
            overall_density  = density
            overall_offset   = offset
            overall_inserted = inserted
            overall_sheet    = sheet
            if density > density_escape: break
            
    toc = time_function()       
    log.info('Packing finished %i iterations in %f seconds', i+1, toc-tic)
    log.info('%i/%i parts were packed successfully', np.sum(overall_inserted), len(polygons))
    log.info('Final rectangular density is %f.', overall_density)
    
    polygon_density = np.sum(polygon_area[overall_inserted])/np.product(overall_sheet)
    log.info('Final polygonal density is %f.', polygon_density)

    transforms_obb    = transforms_obb[overall_inserted]
    transforms_packed = transforms_obb.copy()
    transforms_packed.reshape(-1,9)[:,[2,5]] += overall_offset + buffer_dist
 
    if plot: 
        transform_polygon(np.array(polygons)[overall_inserted],  
                          transforms_packed, 
                          plot=True)
        plt.show()

    rectangles -= 2.0*buffer_dist
    
    if return_all:
        return (overall_inserted, 
                transforms_packed, 
                transforms_obb, 
                overall_sheet,
                rectangles[overall_inserted])

    return overall_inserted, transforms_packed

class Packer:
    def __init__(self, sheet_size=None):
        pass

    def add(self):
        pass
