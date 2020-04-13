"""
material.py
-------------

Store visual materials as objects.
"""
import numpy as np

from . import color
from .. import util


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

    def to_obj(self, tex_name=None, mtl_name=None):
        """
        Convert the current material to an OBJ format material.

        Parameters
        -----------
        name : str or None
          Name to apply to the material

        Returns
        -----------
        tex_name : str
          Name of material
        mtl_name : str
          Name of mtl file in files
        files : dict
          Data as {file name : bytes}
        """
        # material parameters as 0.0-1.0 RGB
        Ka = color.to_float(self.ambient)[:3]
        Kd = color.to_float(self.diffuse)[:3]
        Ks = color.to_float(self.specular)[:3]

        if tex_name is None:
            tex_name = 'material0'
        if mtl_name is None:
            mtl_name = '{}.mtl'.format(tex_name)

        # what is the name of the export image to save
        image_name = '{}.{}'.format(
            tex_name, self.image.format).lower()

        # create an MTL file
        mtl = '\n'.join(
            ['# https://github.com/mikedh/trimesh',
             'newmtl {}'.format(tex_name),
             'Ka {} {} {}'.format(*Ka),
             'Kd {} {} {}'.format(*Kd),
             'Ks {} {} {}'.format(*Ks),
             'Ns {}'.format(self.glossiness),
             'map_Kd {}'.format(image_name)])

        # save the image texture as bytes in the original format
        f_obj = util.BytesIO()
        self.image.save(fp=f_obj, format=self.image.format)
        f_obj.seek(0)

        # collect the OBJ data into files
        data = {mtl_name: mtl.encode('utf-8'),
                image_name: f_obj.read()}

        return data, tex_name, mtl_name

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
        self.name = name
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
