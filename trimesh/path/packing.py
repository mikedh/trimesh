"""
packing.py
------------

Pack multiple 2D regions onto larger 2D regions.
"""
import time
import numpy as np

from ..constants import log

from scipy.spatial import Rectangle

# floating point zero
_TOL_ZERO = 1e-12


class RectangleBin:
    """
    An N-dimensional BSP tree node for packing
    hyper-rectangles using scipy.spatial.Rectangle

    Mostly useful for packing 2D textures and 3D boxes
    http://www.blackpawn.com/texts/lightmaps/
    """

    def __init__(self, bounds):
        """
        Create a rectangular bin.

        Parameters
        ------------
        bounds : (dimension * 2,) float or scipy.spatial.Rectangle
          Bounds array are mins, maxes: [minx, miny, maxx, maxy]
        """
        # this is a *binary* tree so regardless of the dimensionality
        # of the rectangles each node has exactly two children
        self.child = [None, None]
        # is this node occupied.
        self.occupied = False

        # accept either `Rectangle` objects or bounds.
        if isinstance(bounds, Rectangle):
            self.rectangle = bounds
        else:
            # assume bounds are a list
            self.rectangle = Rectangle(
                *np.array(bounds, dtype=np.float64).reshape((2, -1)))

    @property
    def extents(self):
        """
        Bounding box size.

        Returns
        ----------
        extents : (2,) float
          Edge lengths of bounding box
        """
        rect = self.rectangle
        return rect.maxes - rect.mins

    def insert(self, size):
        """
        Insert a rectangle into the bin.

        Parameters
        -------------
        size : (dimension,) float
          Size of rectangle to insert

        Returns
        ----------
        inserted : None or (2,) float
          Position of insertion in the tree
        """
        for child in self.child:
            if child is not None:
                # try inserting into child cells
                attempt = child.insert(size=size)
                if attempt is not None:
                    return attempt

        # can't insert into occupied cells
        if self.occupied:
            return None

        current = self.rectangle
        extents = current.maxes - current.mins

        # compare the bin size to the insertion candidate size
        # manually compute extents here to avoid function call
        size_test = extents - size

        # this means the inserted rectangle is too big for the cell
        if (size_test < -_TOL_ZERO).any():
            return None

        # since the cell is big enough for the current rectangle, either it
        # is going to be inserted here, or the cell is going to be split
        # either way the cell is now occupied.
        self.occupied = True

        # this means the inserted rectangle fits perfectly
        # since we already checked to see if it was negative
        # no abs is needed
        if (size_test < _TOL_ZERO).all():
            # print('got it!')
            return self.rectangle.mins

        # pick the split axis
        axis = size_test.argmax()
        # split hyper-rectangle along axis
        # note that split is *absolute* distance not offset
        # so we have to add the current min to the size
        split = current.split(
            d=axis, split=current.mins[axis] + size[axis])

        # assign two children
        self.child[:] = RectangleBin(split[0]), RectangleBin(split[1])

        # insert the requested box
        return self.child[0].insert(size)


def rectangles_single(rectangles, sheet_size=None, shuffle=False):
    """
    Execute a single insertion order of smaller rectangles onto
    a larger rectangle using a binary space partition tree.

    Parameters
    ----------
    rectangles : (n, 2) float
      An array of (width, height) pairs
      representing the rectangles to be packed.
    sheet_size : (2,) float
      Width, height of rectangular sheet
    shuffle : bool
      Whether or not to shuffle the insert order of the
      smaller rectangles, as the final packing density depends
      on insertion order.

    Returns
    ---------
    density : float
      Area filled over total sheet area
    offset :  (m,2) float
      Offsets to move rectangles to their packed location
    inserted : (n,) bool
      Which of the original rectangles were packed
    consumed_box : (2,) float
      Bounding box size of packed result
    """
    rectangles = np.asanyarray(rectangles, dtype=np.float64)
    dimension = rectangles.shape[1]

    offset = np.zeros((len(rectangles), dimension))
    inserted = np.zeros(len(rectangles), dtype=bool)
    box_order = np.argsort(np.sum(rectangles**2, axis=1))[::-1]
    area = 0.0
    density = 0.0

    # if no sheet size specified, make a large one
    if sheet_size is None:
        sheet_size = np.ones(dimension) * rectangles.sum(
            axis=0).max() * 1.1

    if shuffle:
        # maximum index to shuffle
        max_idx = int(np.random.random() * len(rectangles)) - 1
        # reorder with permutations
        box_order[:max_idx] = np.random.permutation(box_order[:max_idx])

    # start the tree
    sheet = RectangleBin(bounds=[np.zeros(len(sheet_size)), sheet_size])
    for index in box_order:
        insert_location = sheet.insert(rectangles[index])
        if insert_location is not None:
            area += np.prod(rectangles[index])
            offset[index] += insert_location
            inserted[index] = True
    consumed_box = np.max((offset + rectangles)[inserted], axis=0)
    density = area / np.product(consumed_box)

    return density, offset[inserted], inserted, consumed_box


def paths(paths, **kwargs):
    """
    Pack a list of Path2D objects into a rectangle.

    Parameters
    ------------
    paths: (n,) Path2D
      Geometry to be packed

    Returns
    ------------
    packed : trimesh.path.Path2D
      Object containing input geometry
    inserted : (m,) int
      Indexes of paths inserted into result
    """
    from .util import concatenate

    # default quantity to 1
    quantity = [i.metadata.get('quantity', 1)
                for i in paths]

    # pack using exterior polygon (will OBB)
    packable = [i.polygons_closed[i.root[0]] for i in paths]

    # pack the polygons using rectangular bin packing
    inserted, transforms = polygons(polygons=packable,
                                    quantity=quantity,
                                    **kwargs)

    multi = []
    for i, T in zip(inserted, transforms):
        multi.append(paths[i].copy())
        multi[-1].apply_transform(T)
    # append all packed paths into a single Path object
    packed = concatenate(multi)

    return packed, inserted


def polygons(polygons,
             sheet_size=None,
             iterations=50,
             density_escape=.95,
             spacing=0.094,
             quantity=None,
             **kwargs):
    """
    Pack polygons into a rectangle by taking each Polygon's OBB
    and then packing that as a rectangle.

    Parameters
    ------------
    polygons : (n,) shapely.geometry.Polygon
      Source geometry
    sheet_size : (2,) float
      Size of rectangular sheet
    iterations : int
      Number of times to run the loop
    density_escape : float
      When to exit early (0.0 - 1.0)
    spacing : float
      How big a gap to leave between polygons
    quantity : (n,) int, or None
      Quantity of each Polygon

    Returns
    -------------
    overall_inserted : (m,) int
      Indexes of inserted polygons
    packed : (m, 3, 3) float
      Homogeonous transforms from original frame to
      packed frame.
    """

    from .polygons import polygons_obb

    if quantity is None:
        quantity = np.ones(len(polygons), dtype=np.int64)
    else:
        quantity = np.asanyarray(quantity, dtype=np.int64)
    if len(quantity) != len(polygons):
        raise ValueError('quantity must match polygons')

    # find the oriented bounding box of the polygons
    obb, rect = polygons_obb(polygons)

    # pad all sides of the rectangle
    rect += 2.0 * spacing
    # move the OBB transform so the polygon is centered
    # in the padded rectangle
    for i, r in enumerate(rect):
        obb[i][:2, 2] += r * .5

    # for polygons occurring multiple times
    indexes = np.hstack([np.ones(q, dtype=np.int64) * i
                         for i, q in enumerate(quantity)])
    # stack using advanced indexing
    obb = obb[indexes]
    rect = rect[indexes]

    # store timing
    tic = time.time()
    density = 0.0

    # run packing for a number of iterations
    (density,
     offset,
     inserted,
     sheet) = rectangles(
         rectangles=rect,
         sheet_size=sheet_size,
         spacing=spacing,
         density_escape=density_escape,
         iterations=iterations, **kwargs)

    toc = time.time()
    log.debug('packing finished %i iterations in %f seconds',
              i + 1,
              toc - tic)
    log.debug('%i/%i parts were packed successfully',
              np.sum(inserted),
              quantity.sum())

    # transformations to packed positions
    packed = obb[inserted]
    # apply the offset and inter- polygon spacing
    packed.reshape(-1, 9)[:, [2, 5]] += offset + spacing

    return indexes[inserted], packed


def rectangles(rectangles,
               sheet_size=None,
               density_escape=0.9,
               spacing=0.0,
               iterations=50,
               quanta=None):
    """
    Run multiple iterations of rectangle packing.

    Parameters
    ------------
    rectangles : (n, 2) float
      Size of rectangles to be packed
    sheet_size : None or (2,) float
      Size of sheet to pack onto
    density_escape : float
      Exit early if density is above this threshold
    spacing : float
      Distance to allow between rectangles
    iterations : int
      Number of iterations to run
    quanta : None or float


    Returns
    ---------
    density : float
      Area filled over total sheet area
    offset :  (m,2) float
      Offsets to move rectangles to their packed location
    inserted : (n,) bool
      Which of the original rectangles were packed
    consumed_box : (2,) float
      Bounding box size of packed result
    """
    rectangles = np.array(rectangles)

    # best density percentage in 0.0 - 1.0
    best_density = 0.0
    # how many rectangles were inserted
    best_insert = 0

    for i in range(iterations):
        # run a single insertion order
        # don't shuffle the first run, shuffle subsequent runs
        packed = rectangles_single(
            rectangles,
            sheet_size=sheet_size,
            shuffle=(i != 0))
        density = packed[0]
        insert = packed[2].sum()

        if quanta is not None:
            # compute the density using an upsized quanta
            box = np.ceil(packed[3] / quanta) * quanta
            # scale the density result
            density *= (np.product(packed[3]) / np.product(box))

        # compare this packing density against our best
        if density > best_density or insert > best_insert:
            best_density = density
            best_insert = insert
            # save the result
            result = packed
            # exit early if everything is inserted and
            # we have exceeded our target density
            if density > density_escape and packed[2].all():
                break

    return result


def images(images, power_resize=False):
    """
    Pack a list of images and return result and offsets.

    Parameters
    ------------
    images : (n,) PIL.Image
      Images to be packed
    deduplicate : bool
      If True deduplicate images before packing

    Returns
    -----------
    packed : PIL.Image
      Multiple images packed into result
    offsets : (n, 2) int
       Offsets for original image to pack
    """
    from PIL import Image

    # use the number of pixels as the rectangle size
    rect = np.array([i.size for i in images])

    (density,
     offset,
     insert,
     sheet) = rectangles(rectangles=rect)
    # really should have inserted all the rectangles
    assert insert.all()

    # offsets should be integer multiple of pizels
    offset = offset.round().astype(int)

    size = sheet.round().astype(int)
    if power_resize:
        # round up all dimensions to powers of 2
        size = (2 ** np.ceil(np.log2(size))).astype(np.int64)

    # create the image
    result = Image.new('RGB', tuple(size))
    # paste each image into the result
    for img, off in zip(images, offset):
        result.paste(img, tuple(off))

    return result, offset
