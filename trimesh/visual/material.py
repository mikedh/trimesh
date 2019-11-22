"""
material.py
-------------

Store visual materials as objects.
"""
import numpy as np

from . import color


class Material(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('material must be subclassed!')

    def __hash__(self):
        return id(self)

    @property
    def main_color(self):
        raise NotImplementedError('material must be subclassed!')


class SimpleMaterial(Material):
    """
    Hold a single image texture.
    """

    def __init__(self,
                 image=None,
                 diffuse=None,
                 ambient=None,
                 specular=None,
                 glossiness=None,
                 **kwargs):

        # save image
        self.image = image

        # save material colors as RGBA
        self.ambient = color.to_rgba(ambient)
        self.diffuse = color.to_rgba(diffuse)
        self.specular = color.to_rgba(specular)

        # save Ns
        self.glossiness = glossiness

        # save other keyword arguments
        self.kwargs = kwargs

    def to_color(self, uv):
        return color.uv_to_color(uv, self.image)

    def __hash__(self):
        """
        Provide a hash of the material so we can detect
        duplicates.

        Returns
        ------------
        hash : int
          Hash of image and parameters
        """
        if hasattr(self.image, 'tobytes'):
            # start with hash of raw image bytes
            hashed = hash(self.image.tobytes())
        else:
            # otherwise start with zero
            hashed = 0
        # we will add additional parameters with
        # an in-place xor of the additional value
        # if stored as numpy arrays add parameters
        if hasattr(self.ambient, 'tobytes'):
            hashed ^= hash(self.ambient.tobytes())
        if hasattr(self.diffuse, 'tobytes'):
            hashed ^= hash(self.diffuse.tobytes())
        if hasattr(self.specular, 'tobytes'):
            hashed ^= hash(self.specular.tobytes())
        if isinstance(self.glossiness, float):
            hashed ^= hash(int(self.glossiness * 1000))
        return hashed

    @property
    def main_color(self):
        """
        Return the most prominent color.
        """
        return self.diffuse

    @property
    def glossiness(self):
        if hasattr(self, '_glossiness'):
            return self._glossiness
        return 1.0

    @glossiness.setter
    def glossiness(self, value):
        if value is None:
            return
        self._glossiness = float(value)

    def to_pbr(self):
        """
        Convert the current simple material to a PBR material.

        Returns
        ------------
        pbr : PBRMaterial
          Contains material information in PBR format.
        """
        # convert specular exponent to roughness
        roughness = (2 / (self.glossiness + 2)) ** (1.0 / 4.0)

        return PBRMaterial(roughnessFactor=roughness,
                           baseColorTexture=self.image,
                           baseColorFactor=self.diffuse)


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

        # (4,) float
        if baseColorFactor is not None:
            baseColorFactor = color.to_rgba(baseColorFactor)
        self.baseColorFactor = baseColorFactor

        if emissiveFactor is not None:
            emissiveFactor = np.array(emissiveFactor, dtype=np.float64)

        # (3,) float
        self.emissiveFactor = emissiveFactor

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
        self.alphaMode = alphaMode

    def to_color(self, uv):
        """
        Get the rough color at a list of specified UV coordinates.

        Parameters
        -------------
        uv : (n, 2) float
          UV coordinates on the material

        Returns
        -------------
        colors :
        """
        colors = color.uv_to_color(uv=uv, image=self.baseColorTexture)
        if colors is None and self.baseColorFactor is not None:
            colors = self.baseColorFactor.copy()
        return colors

    @property
    def main_color(self):
        # will return default color if None
        result = color.to_rgba(self.baseColorFactor)
        return result
