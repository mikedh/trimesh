"""
material.py
-------------

Store visual materials as objects.
"""
import copy
import numpy as np

from . import color
from .. import util
from .. import grouping


class Material(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('material must be subclassed!')

    def __hash__(self):
        return id(self)

    @property
    def main_color(self):
        raise NotImplementedError('material must be subclassed!')

    @property
    def name(self):
        if hasattr(self, '_name'):
            return self._name
        return 'material0'

    @name.setter
    def name(self, value):
        self._name = value

    def copy(self):
        return copy.deepcopy(self)


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

    def to_obj(self, mtl_name=None):
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

        # get the name of this material
        tex_name = self.name
        if mtl_name is None:
            mtl_name = '{}.mtl'.format(tex_name)

        # create an MTL file
        mtl = ['# https://github.com/mikedh/trimesh',
               'newmtl {}'.format(tex_name),
               'Ka {:0.8f} {:0.8f} {:0.8f}'.format(*Ka),
               'Kd {:0.8f} {:0.8f} {:0.8f}'.format(*Kd),
               'Ks {:0.8f} {:0.8f} {:0.8f}'.format(*Ks),
               'Ns {:0.8f}'.format(self.glossiness)]

        # collect the OBJ data into files
        data = {}

        if self.image is not None:
            image_type = self.image.format
            # what is the name of the export image to save
            if image_type is None:
                image_type = 'png'
            image_name = '{}.{}'.format(
                tex_name, image_type).lower()
            # save the reference to the image
            mtl.append('map_Kd {}'.format(image_name))

            # save the image texture as bytes in the original format
            f_obj = util.BytesIO()
            self.image.save(fp=f_obj, format=image_type)
            f_obj.seek(0)
            data[image_name] = f_obj.read()

        data[mtl_name] = '\n'.join(mtl).encode('utf-8')

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

    def copy(self):
        # doing a straight deepcopy fails probably due to PIL images
        kwargs = {}
        # collect stored values as kwargs
        for k, v in self.__dict__.items():
            if v is None:
                continue
            if k.startswith('_'):
                k = k[1:]
            if hasattr(v, 'copy'):
                # use an objects explicit copy if available
                kwargs[k] = v.copy()
            else:
                # otherwise just hope deepcopy does something
                kwargs[k] = copy.deepcopy(v)
        return PBRMaterial(**kwargs)

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

    def to_simple(self):
        """
        Get a copy of the current PBR material as a simple material.

        Returns
        ------------
        simple : SimpleMaterial
          Contains material information in a simple manner
        """

        return SimpleMaterial(image=self.baseColorTexture,
                              diffuse=self.baseColorFactor)

    @property
    def main_color(self):
        # will return default color if None
        result = color.to_rgba(self.baseColorFactor)
        return result

    def __hash__(self):
        """
        Provide a hash of the material so we can detect
        duplicate materials.

        Returns
        ------------
        hash : int
          Hash of image and parameters
        """
        if hasattr(self.baseColorTexture, 'tobytes'):
            # start with hash of raw image bytes
            hashed = hash(self.baseColorTexture.tobytes())
        else:
            # otherwise start with zero
            hashed = 0
        # we will add additional parameters with
        # an in-place xor of the additional value
        # if stored as numpy arrays add parameters
        if hasattr(self.baseColorFactor, 'tobytes'):
            hashed ^= hash(self.baseColorFactor.tobytes())

        return hashed


def empty_material(color=None):
    """
    Return an empty material set to a single color

    Parameters
    -----------
    color : None or (3,) uint8
      RGB color

    Returns
    -------------
    material : SimpleMaterial
      Image is a a one pixel RGB
    """
    from PIL import Image
    if color is None or np.shape(color) not in ((3,), (4,)):
        color = np.array([255, 255, 255], dtype=np.uint8)
    else:
        color = np.array(color, dtype=np.uint8)[:3]
    # create a one pixel RGB image
    image = Image.fromarray(
        np.tile(color, (4, 1)).reshape((2, 2, 3)))
    return SimpleMaterial(image=image)


def from_color(vertex_colors):
    """
    Convert vertex colors into UV coordinates and materials.

    TODO : pack colors

    Parameters
    ------------
    vertex_colors : (n, 3) float
      Array of vertex colors

    Returns
    ------------
    material : SimpleMaterial
      Material containing color information
    uvs : (n, 2) float
      UV coordinates
    """
    unique, inverse = grouping.unique_rows(vertex_colors)
    # TODO : tile colors nicely
    material = empty_material(color=vertex_colors[unique[0]])
    uvs = np.zeros((len(vertex_colors), 2)) + 0.5

    return material, uvs


def pack(materials, uvs, deduplicate=True):
    """
    Pack multiple materials with texture into a single material.

    Parameters
    -----------
    materials : (n,) Material
      List of multiple materials
    uvs : (n, m, 2) float
      Original UV coordinates

    Returns
    ------------
    material : Material
      Combined material
    uv : (p, 2) float
      Combined UV coordinates
    """

    from PIL import Image
    from ..path import packing
    import collections

    if deduplicate:
        # start by collecting a list of indexes for each material hash
        unique_idx = collections.defaultdict(list)
        [unique_idx[hash(m)].append(i) for i, m in enumerate(materials)]
        # now we only need the indexes and don't care about the hashes
        mat_idx = list(unique_idx.values())
    else:
        # otherwise just use all the indexes
        mat_idx = np.arange(len(materials)).reshape((-1, 1))

    # store the images to combine later
    images = []
    # first collect the images from the materials
    for idx in mat_idx:
        # get the first material from the group
        m = materials[idx[0]]
        # extract an image for each material
        if isinstance(m, PBRMaterial):
            if m.baseColorTexture is not None:
                img = m.baseColorTexture
            elif m.baseColorFactor is not None:
                img = Image.fromarray(m.baseColorFactor[:3].reshape((1, 1, 3)))
            else:
                img = Image.new(mode='RGB', size=(1, 1))
        elif hasattr(m, 'image'):
            img = m.image
        else:
            raise ValueError('no image to pack!')
        images.append(img)

    # pack the multiple images into a single large image
    final, offsets = packing.images(images, power_resize=True)

    # the size of the final texture image
    final_size = np.array(final.size, dtype=np.float64)
    # collect scaled new UV coordinates
    new_uv = []

    for idxs, img, off in zip(mat_idx, images, offsets):
        # how big was the original image
        scale = img.size / final_size
        # what is the offset in fractions of final image
        uv_off = off / final_size
        # scale and translate each of the new UV coordinates
        # [new_uv.append((uvs[i] * scale) + uv_off) for i in idxs]
        # TODO : figure out why this is broken sometimes...
        [new_uv.append((uvs[i] * scale) + uv_off) for i in idxs]

    # stack UV coordinates into single (n, 2) array
    stacked = np.vstack(new_uv)

    return SimpleMaterial(image=final), stacked
