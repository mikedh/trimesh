import numpy as np

import copy

from . import color
from .. import caching


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


class Material(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('material must be subclassed!')


class SimpleMaterial(Material):
    """
    Hold a single image texture.
    """

    def __init__(self,
                 image=None,
                 diffuse=None,
                 ambient=None,
                 specular=None):

        # save image
        self.image = image

        # save material colors
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

    def to_color(self, uv):
        return uv_to_color(uv, self.image)


class PBRMaterial(Material):
    """
    Create a material for physically based rendering as
    specified by GLTF 2.0:
    https://git.io/fhkPZ

    Parameters with `Texture` in them must be PIL.Image objects
    """

    def __init__(self,
                 name=None,
                 emissiveFactor=None,
                 emissiveTexture=None,
                 normalTexture=None,
                 occlusionTexture=None,
                 baseColorTexture=None,
                 baseColorFactor=None,
                 metallicFactor=None,
                 roughnessFactor=None,
                 metallicRoughnessTexture=None,
                 doubleSided=False,
                 alphaMode='OPAQUE',
                 alphaCutoff=0.5):

        # To to-float conversions
        if baseColorFactor is not None:
            baseColorFactor = np.array(baseColorFactor, dtype=np.float)
        if emissiveFactor is not None:
            emissiveFactor = np.array(emissiveFactor, dtype=np.float)

        # (4,) float
        self.baseColorFactor = color.to_rgba(baseColorFactor)

        # (3,) float
        self.emissiveFactor = color.to_rgba(emissiveFactor)

        # float
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor
        self.alphaCutoff = alphaCutoff

        # PIL image
        self.normalTexture = normalTexture
        self.emissiveTexture = emissiveTexture
        self.occlusionTexture = occlusionTexture
        self.baseColorTexture = baseColorTexture
        self.metallicRoughnessTexture = metallicRoughnessTexture

        # bool
        self.doubleSided = doubleSided

        # str
        alphaMode = alphaMode

    def to_color(self, uv):
        color = uv_to_color(uv=uv, image=self.baseColorTexture)
        if color is None and self.baseColorFactor is not None:
            color = self.baseColorFactor.copy()
        return color


def uv_to_color(uv, image):
    """
    Get the color in a texture image.

    Parameters
    -------------
    uv : (n, 2) float
      UV coordinates on texture image
    image : PIL.Image
      Texture image

    Returns
    ----------
    colors : (n, 4) float
      RGBA color at each of the UV coordinates
    """
    if image is None or uv is None:
        return None

    # UV coordinates should be (n, 2) float
    uv = np.asanyarray(uv, dtype=np.float64)

    # get texture image pixel positions of UV coordinates
    x = (uv[:, 0] * (image.width - 1))
    y = ((1 - uv[:, 1]) * (image.height - 1))

    # convert to int and wrap to image
    # size in the manner of GL_REPEAT
    x = x.round().astype(np.int64) % image.width
    y = y.round().astype(np.int64) % image.height

    # access colors from pixel locations
    # make sure image is RGBA before getting values
    colors = np.asanyarray(image.convert('RGBA'))[y, x]

    # conversion to RGBA should have corrected shape
    assert colors.ndim == 2 and colors.shape[1] == 4

    return colors
