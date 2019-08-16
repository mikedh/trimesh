import numpy as np

import copy

from . import color

from .. import caching
from .. import grouping

from .material import SimpleMaterial, PBRMaterial  # NOQA


class TextureVisuals(object):
    def __init__(self,
                 uv=None,
                 material=None,
                 image=None):
        """
        Store a single material and per-vertex UV coordinates
        for a mesh.

        If passed UV coordinates and a single image it will
        create a SimpleMaterial for the image.

        Parameters
        --------------
        uv : (n, 2) float
          UV coordinates for the mesh
        material : Material
          Store images and properties
        image : PIL.Image
          Can be passed to automatically create material
        """

        # store values we care about enough to hash
        self._data = caching.DataStore()
        # cache calculated values
        self._cache = caching.Cache(self._data.fast_hash)

        # should be (n, 2) float
        self.uv = uv

        if material is None and image is not None:
            # if an image is passed create a SimpleMaterial
            self.material = SimpleMaterial(image=image)
        else:
            # may be None
            self.material = material

    def _verify_crc(self):
        """
        Dump the cache if anything in self._data has changed.
        """
        self._cache.verify()

    @property
    def kind(self):
        """
        Return the type of visual data stored

        Returns
        ----------
        kind : str
          What type of visuals are defined
        """
        return 'texture'

    @property
    def defined(self):
        """
        Check if any data is stored

        Returns
        ----------
        defined : bool
          Are UV coordinates and images set?
        """
        ok = self.material is not None
        return ok

    def crc(self):
        """
        Get a CRC of the stored data.

        Returns
        --------------
        crc : int
          Hash of items in self._data
        """
        return self._data.crc()

    @property
    def uv(self):
        """
        Get the stored UV coordinates.

        Returns
        ------------
        uv : (n, 2) float
          Pixel position per- vertex
        """
        if 'uv' in self._data:
            return self._data['uv']
        return None

    @uv.setter
    def uv(self, values):
        """
        Set the UV coordinates.

        Parameters
        --------------
        values : (n, 2) float
          Pixel locations on a texture per- vertex
        """
        if values is None:
            self._data.clear()
        else:
            self._data['uv'] = np.asanyarray(values, dtype=np.float64)

    def copy(self):
        """
        Return a copy of the current TextureVisuals object.

        Returns
        ----------
        copied : TextureVisuals
          Contains the same information in a new object
        """
        uv = self.uv
        if uv is not None:
            uv = uv.copy()
        copied = TextureVisuals(
            uv=uv,
            material=copy.deepcopy(self.material))

        return copied

    def to_color(self):
        """
        Convert textured visuals to a ColorVisuals with vertex
        color calculated from texture.

        Returns
        -----------
        vis : trimesh.visuals.ColorVisuals
          Contains vertex color from texture
        """
        # find the color at each UV coordinate
        colors = self.material.to_color(self.uv)
        # create ColorVisuals from result
        vis = color.ColorVisuals(vertex_colors=colors)
        return vis

    def face_subset(self, face_index):
        """
        Get a copy of
        """
        return self.copy()

    def update_vertices(self, mask):
        """
        Apply a mask to remove or duplicate vertex properties.
        """
        if self.uv is not None:
            self.uv = self.uv[mask]

    def update_faces(self, mask):
        """
        Apply a mask to remove or duplicate face properties
        """
        pass


def unmerge_faces(faces, *args):
    """
    Textured meshes can come with faces referencing vertex
    indices (`v`) and an array the same shape which references
    vertex texture indices (`vt`) and sometimes even normal (`vn`).

    Vertex locations with different values of any of these can't
    be considered the "same" vertex, and for our simple data
    model we need to not combine these vertices.

    Parameters
    -------------
    faces : (n, d) int
      References vertex indices
    *args : (n, d) int
      Various references of corresponding values
      This is usually UV coordinates or normal indexes

    Returns
    -------------
    new_faces : (m, d) int
      New faces for masked vertices
    mask_v : (p,) int
      A mask to apply to vertices
    mask_* : (p,) int
      A mask to apply to vt array to get matching UV coordinates
      Returns as many of these as args were passed
    """
    # stack into pairs of (vertex index, texture index)
    stackable = [np.asanyarray(faces).reshape(-1)]
    # append multiple args to the correlated stack
    # this is usually UV coordinates (vt) and normals (vn)
    for arg in args:
        stackable.append(np.asanyarray(arg).reshape(-1))
    # unify them into rows of a numpy array
    stack = np.column_stack(stackable)
    # find unique pairs: we're trying to avoid merging
    # vertices that have the same position but different
    # texture coordinates
    unique, inverse = grouping.unique_rows(stack)

    # only take the unique pairs
    pairs = stack[unique]
    # try to maintain original vertex order
    order = pairs[:, 0].argsort()
    # apply the order to the pairs
    pairs = pairs[order]

    # we re-ordered the vertices to try to maintain
    # the original vertex order as much as possible
    # so to reconstruct the faces we need to remap
    remap = np.zeros(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))

    # the faces are just the inverse with the new order
    new_faces = remap[inverse].reshape((-1, 3))

    # the mask for vertices and masks for other args
    result = [new_faces]
    result.extend(pairs.T)

    return result


def power_resize(image, resample=1, square=False):
    """
    Resize a PIL image so every dimension is a power of two.

    Parameters
    ------------
    image : PIL.Image
      Input image
    resample : int
      Passed to Image.resize
    square : bool
      If True, upsize to a square image

    Returns
    -------------
    resized : PIL.Image
      Input image resized
    """
    # what is the current resolution of the image in pixels
    size = np.array(image.size, dtype=np.int64)
    # what is the resolution of the image upsized to the nearest
    # power of two on each axis: allow rectangular textures
    new_size = (2 ** np.ceil(np.log2(size))).astype(np.int64)

    # make every dimension the largest
    if square:
        new_size = np.ones(2, dtype=np.int64) * new_size.max()

    # if we're not powers of two upsize
    if (size != new_size).any():
        return image.resize(new_size, resample=resample)

    return image.copy()
