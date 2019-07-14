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
                 **kwargs):

        # save image
        self.image = image

        # save material colors as RGBA
        self.ambient = color.to_rgba(ambient)
        self.diffuse = color.to_rgba(diffuse)
        self.specular = color.to_rgba(specular)

        # save other keyword arguments
        self.kwargs = kwargs

    def to_color(self, uv):
        return color.uv_to_color(uv, self.image)

    @property
    def main_color(self):
        return self.diffuse


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
        colors = color.uv_to_color(uv=uv, image=self.baseColorTexture)
        if colors is None and self.baseColorFactor is not None:
            colors = self.baseColorFactor.copy()
        return colors

    @property
    def main_color(self):
        return self.baseColorFactor
