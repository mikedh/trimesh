import numpy as np

from collections import deque

from ..constants import log, time_function
from ..constants import tol_path as tol

from .polygons import polygons_obb, transform_polygon


class RectangleBin:
    '''
    2D BSP tree node.
    http://www.blackpawn.com/texts/lightmaps/
    '''

    def __init__(self, bounds=None, size=None):
        self.child = [None] * 2

        # bounds: (minx, miny, maxx, maxy)
        self.bounds = bounds
        self.occupied = False

        if size is not None:
            self.bounds = np.append([0, 0], size)

    def insert(self, rectangle_size):
        for child in self.child:
            if child is not None:
                attempt = child.insert(rectangle_size)
                if attempt:
                    return attempt

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
        vertical = size_test[0] > size_test[1]
        length = rectangle_size[int(not vertical)]
        child_bounds = self.split(length, vertical)

        self.child[0] = RectangleBin(bounds=child_bounds[0])
        self.child[1] = RectangleBin(bounds=child_bounds[1])

        return self.child[0].insert(rectangle_size)

    def split(self, length=None, vertical=True):
        '''
        returns two bounding boxes representing the current
        bounds split into two smaller boxes
        '''
        # also know as [minx, miny, maxx, maxy]
        [left, bottom, right, top] = self.bounds
        if vertical:
            box = [[left, bottom, left + length, top],
                   [left + length, bottom, right, top]]
        else:
            box = [[left, bottom, right, bottom + length],
                   [left, bottom + length, right, top]]
        return box


def bounds_to_size(bounds):
    return np.diff(np.reshape(bounds, (2, 2)), axis=0)[0]


def pack_rectangles(rectangles, sheet_size, shuffle=False):
    '''
    Pack smaller rectangles onto a larger rectangle, using a binary
    space partition tree.

    Parameters
    ----------
    rectangles: (n,2) array of (width, height) pairs
                 representing the smaller rectangles to be packed.
    sheet_size: (2) array of (width, height) pair representing
                 the sheet size the smaller rectangles will be packed onto.
    shuffle: boolean, whether or not to shuffle the insert order of the
                 smaller rectangles, as the final packing density depends on the
                 order of which rectangles are inserted onto the larger sheet.

    Returns
    ---------
    density: float, effective density
    offset: (m,2) float, offsets to packed location
    inserted: (n,) bool, which of the original rectangles were packed
    consumed_box: (2,) bounding box of resulting packing
    '''
    offset = np.zeros((len(rectangles), 2))
    inserted = np.zeros(len(rectangles), dtype=np.bool)
    box_order = np.argsort(np.sum(rectangles**2, axis=1))[::-1]
    area = 0.0
    density = 0.0

    if shuffle:
        shuffle_len = int(np.random.random() * len(rectangles)) - 1
        box_order[0:shuffle_len] = np.random.permutation(
            box_order[0:shuffle_len])

    sheet = RectangleBin(size=sheet_size)
    for index in box_order:
        insert_location = sheet.insert(rectangles[index])
        if insert_location is not None:
            area += np.prod(rectangles[index])
            offset[index] += insert_location
            inserted[index] = True

    consumed_box = np.max((offset + rectangles)[inserted], axis=0)
    density = area / np.product(consumed_box)

    return density, offset[inserted], inserted, consumed_box


def pack_paths(paths, sheet_size=None):
    """
    Pack a list of Path2D objects into a rectangle.

    Parameters
    ------------
    paths: (n,) list, of Path2D objects

    Returns
    ------------
    packed: Path2D object
    """    
    multi = []
    for path in paths:
        if 'quantity' in path.metadata:
            count = path.metadata['quantity']
        else:
            count = 1
        for i in range(path.metadata['quantity']):
            multi.append(path.copy())

    polygons = [i.polygons_closed[i.root[0]] for i in multi]
    inserted, transforms = multipack(polygons=polygons,
                                     sheet_size=sheet_size)

    for path, transform in zip(multi, transforms):
        path.apply_transform(transform)
    packed = sum(multi)

    return packed


def multipack(polygons,
              sheet_size=None,
              iterations=50,
              density_escape=.95,
              spacing=0.125):
    """
    Pack polygons into a rectangle.

    Parameters
    ------------
    polygons:   (n,) list, of shapely.geometry.Polygon objects
    sheet_size: (2,) float, size of sheet
    iterations: int, number of times to run the loop
    density_escape: float, when to exit early
    spacing:        float, how big a gap to leave between polygons
  
    Returns
    -------------
    overall_inserted:  (n,) bool, was polygon inserted
    transforms_packed: (m, 3, 3) float, transformations
    """

    # find the oriented bounding box of the polygons
    transforms_obb, rectangles = polygons_obb(polygons)
    # pad all sides of the rectangle
    rectangles += 2.0 * spacing

    # move the OBB transform so the polygon is centered
    # in the padded rectangle
    for i, r in enumerate(rectangles):
        transforms_obb[i][0:2, 2] += r * .5

    tic = time_function()
    overall_density = 0

    # if no sheet size specified, make a large one 
    if sheet_size is None:
        max_dim = np.max(rectangles, axis=0)
        sum_dim = np.sum(rectangles, axis=0)
        sheet_size = [sum_dim[0], max_dim[1] * 2]

    log.debug('packing %d polygons', len(polygons))
    # run packing for a number of iterations, shuffling insertion order
    for i in range(iterations):
        (density, 
         offset, 
         inserted, 
         sheet) = pack_rectangles(rectangles, 
                                  sheet_size=sheet_size, 
                                  shuffle=(i != 0))
        if density > overall_density:
            overall_density = density
            overall_offset = offset
            overall_inserted = inserted
            overall_sheet = sheet
            if density > density_escape:
                break

    toc = time_function()
    log.debug('packing finished %i iterations in %f seconds', 
              i + 1, 
              toc - tic)
    log.debug('%i/%i parts were packed successfully',
             np.sum(overall_inserted), 
              len(polygons))
    log.debug('final rectangular density is %f.', overall_density)

    transforms_packed = transforms_obb[overall_inserted]
    transforms_packed.reshape(-1, 9)[:, [2, 5]] += overall_offset + spacing

    return overall_inserted, transforms_packed
