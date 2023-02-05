"""
packing.py
------------

Pack rectangular regions onto larger rectangular regions.
"""
import time
import numpy as np

from ..constants import log

# floating point zero
_TOL_ZERO = 1e-12


class RectangleBin:
    """
    An N-dimensional binary space partition tree for packing
    hyper-rectangles. Split logic is pure `numpy` but behaves
    similarly to `scipy.spatial.Rectangle`.

    Mostly useful for packing 2D textures and 3D boxes and
    has not been tested outside of 2 and 3 dimensions.

    Original article about using this for packing textures:
    http://www.blackpawn.com/texts/lightmaps/
    """

    def __init__(self, bounds):
        """
        Create a rectangular bin.

        Parameters
        ------------
        bounds : (2, dimension *) float
          Bounds array are `[mins, maxes]`
        """
        # this is a *binary* tree so regardless of the dimensionality
        # of the rectangles each node has exactly two children
        self.child = [None, None]
        # is this node occupied.
        self.occupied = False
        # assume bounds are a list
        self.bounds = np.array(bounds, dtype=np.float64)

    @property
    def extents(self):
        """
        Bounding box size.

        Returns
        ----------
        extents : (dimension,) float
          Edge lengths of bounding box
        """
        bounds = self.bounds
        return bounds[1] - bounds[0]

    def insert(self, size):
        """
        Insert a rectangle into the bin.

        Parameters
        -------------
        size : (dimension,) float
          Size of rectangle to insert/

        Returns
        ----------
        inserted : (2,) float or None
          Position of insertion in the tree or None
          if the insertion was unsucessful.
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

        # shortcut for our bounds
        bounds = self.bounds
        extents = bounds[1] - bounds[0]
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
            return bounds

        # pick the axis to split along
        axis = size_test.argmax()
        # split hyper-rectangle along axis
        # note that split is *absolute* distance not offset
        # so we have to add the current min to the size
        splits = np.tile(bounds, (2, 1))
        splits[1:3, axis] = bounds[0][axis] + size[axis]

        # assign two children
        self.child[:] = RectangleBin(splits[:2]), RectangleBin(splits[2:])

        # insert the requested item into the first child
        return self.child[0].insert(size)


def split(bounds, length, axis=0):
    """
    Split a hyper-rectangle along an axis.

    Parameters
    -------------
    bounds : (2, dimension) float
      Minimum and maximum position of an N-dimensional
      axis aligned bounding box.
    length : float
      Length from the minimum along `axis` to split
      the passed `bounds`.
    axis : int
      Which axis of the hyper-rectangle to split along.

    Returns
    ------------
    split : (2, 2, dimension) float
      Two new axis aligned bounding boxes.
    """

    splits = np.tile(bounds, (2, 1))
    splits[1:3, axis] = bounds[0][axis] + length

    return splits.reshape((2, 2, -1))


def rectangles_single(rect, size=None, shuffle=False):
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

    rect = np.asanyarray(rect, dtype=np.float64)
    dim = rect.shape[1]

    offset = np.zeros((len(rect), 2, dim))
    consume = np.zeros(len(rect), dtype=bool)

    # start by ordering them by maximum length
    order = np.argsort(rect.max(axis=1))[::-1]

    if shuffle:
        # maximum index to shuffle
        max_idx = int(np.random.random() * len(rect)) - 1
        # reorder with permutations
        order[:max_idx] = np.random.permutation(order[:max_idx])

    if size is None:
        # if no bounds are passed start it with the maximum size
        # along an axis which will almost certainly require re-rooting
        root_bounds = [[0, 0, 0], rect.max(axis=0)]
    else:
        # restrict the bounds to passed size and disallow re-rooting
        root_bounds = [[0, 0, 0], size]

    # the current root node to insert each rectangle
    root = RectangleBin(bounds=root_bounds)

    for index in order:
        # the current rectangle to be inserted
        rectangle = rect[index]
        # try to insert the hyper-rectangle into children
        inserted = root.insert(rectangle)

        if inserted is None and size is None:
            # we failed to insert into children
            # so we need to create a new parent
            # get the size of the current root node
            bounds = root.bounds
            extents = bounds.ptp(axis=0)
            stack = np.array([extents, rectangle])

            # we are going to combine two hyper-rect
            # so we have `dim` choices on ways to split
            # choose the split that minimizes the new hyper-volume
            # the new AABB is going to be the `max` of the lengths
            # on every dim except one which will be the `sum`
            choices = np.tile(stack.max(axis=0), (len(extents), 1))
            np.fill_diagonal(choices, stack.sum(axis=0))
            # choose the new AABB by which one minimizes hyper-volume
            choice_idx = np.product(choices, axis=1).argmin()
            # we now know the full extent of the AABB
            new_max = bounds[0] + choices[choice_idx]

            # offset the new bounding box corner
            new_min = bounds[0].copy()
            new_min[choice_idx] += extents[choice_idx]

            # original bounds may be stretched
            new_ori_max = np.vstack((bounds[1], new_max)).max(axis=0)
            new_ori_max[choice_idx] = bounds[1][choice_idx]

            assert (new_ori_max >= bounds[1]).all()

            # the bounds containing the original sheet
            bounds_ori = np.array([bounds[0], new_ori_max])
            # the bounds containing the location to insert
            # the new rectangle
            bounds_ins = np.array([new_min, new_max])

            # generate the new root node
            new_root = RectangleBin([bounds[0], new_max])
            # this node has children so it is occupied
            new_root.occupied = True
            # create a bin for both bounds
            new_root.child[0] = RectangleBin(bounds_ori)
            new_root.child[1] = RectangleBin(bounds_ins)

            # insert the original sheet into the new tree
            root_offset = new_root.child[0].insert(bounds.ptp(axis=0))
            # we sized the cells so original tree would fit
            assert root_offset is not None

            # insert the child that didn't fit before into the other child
            child = new_root.child[1].insert(rectangle)
            # since we re-sized the cells to fit insertion should always work
            assert child is not None

            offset[index] = child
            consume[index] = True

            # subsume the existing tree into a new root
            root = new_root

        else:
            offset[index] = inserted
            consume[index] = True

    return offset, consume


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
     sheet) = rect(
         rect=rect,
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


def rectangles(rect,
               size=None,
               density_escape=0.9,
               spacing=0.0,
               iterations=50,
               quanta=None):
    """
    Run multiple iterations of rectangle packing.

    Parameters
    ------------
    rect : (n, 2) float
      Size of rect to be packed
    sheet_size : None or (2,) float
      Size of sheet to pack onto
    density_escape : float
      Exit early if density is above this threshold
    spacing : float
      Distance to allow between rect
    iterations : int
      Number of iterations to run
    quanta : None or float


    Returns
    ---------
    density : float
      Area filled over total sheet area
    offset :  (m,2) float
      Offsets to move rect to their packed location
    inserted : (n,) bool
      Which of the original rect were packed
    consumed_box : (2,) float
      Bounding box size of packed result
    """
    rect = np.array(rect)

    # best density percentage in 0.0 - 1.0
    best_density = 0.0
    # how many rect were inserted
    best_insert = 0

    for i in range(iterations):
        # run a single insertion order
        # don't shuffle the first run, shuffle subsequent runs
        packed = rect_single(
            rect,
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
     sheet) = rect(rect=rect)
    # really should have inserted all the rect
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
