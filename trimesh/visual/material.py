"""
material.py
-------------

Store visual materials as objects.
"""
import abc
import copy
import numpy as np

from . import color
from .. import util
from .. import grouping
from .. import exceptions

# epsilon for comparing floating point
_eps = 1e-5


class Material(util.ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('must be subclassed!')

    def __hash__(self):
        return id(self)

    @abc.abstractproperty
    def main_color(self):
        """
        The "average" color of this material.

        Returns
        ---------
        color : (4,) uint8
          Average color of this material.
        """

    @property
    def name(self):
        if hasattr(self, '_name'):
            return self._name
        return 'material_0'

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

    def to_obj(self, name=None):
        """
        Convert the current material to an OBJ format
        material.

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

        if name is None:
            name = self.name

        # create an MTL file
        mtl = ['newmtl {}'.format(name),
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
            image_name = '{}.{}'.format(name, image_type.lower())
            # save the reference to the image
            mtl.append('map_Kd {}'.format(image_name))

            # save the image texture as bytes in the original format
            f_obj = util.BytesIO()
            self.image.save(fp=f_obj, format=image_type)
            f_obj.seek(0)
            data[image_name] = f_obj.read()

        data['{}.mtl'.format(name)] = '\n'.join(mtl).encode('utf-8')

        return data, name

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
        Convert the current simple material to a
        PBR material.

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


class MultiMaterial(Material):
    def __init__(self, materials=None, **kwargs):
        """
        Wrapper for a list of Materials.

        Parameters
        ----------
        materials : Optional[List[Material]]
            List of materials with which the container to be initialized.
        """
        if materials is None:
            self.materials = []
        else:
            self.materials = materials

    def to_pbr(self):
        """
        TODO : IMPLEMENT
        """
        pbr = [m for m in self.materials
               if isinstance(m, PBRMaterial)]
        if len(pbr) == 0:
            return PBRMaterial()
        return pbr[0]

    def __hash__(self):
        """
        Provide a hash of the multi material so we can detect
        duplicates.

        Returns
        ------------
        hash : int
          Xor hash of the contained materials.
        """
        hashed = int(np.bitwise_xor.reduce(
            [hash(m) for m in self.materials]))

        return hashed

    def __iter__(self):
        return iter(self.materials)

    def __next__(self):
        return next(self.materials)

    def __len__(self):
        return len(self.materials)

    @property
    def main_color(self):
        """
        The "average" color of this material.

        Returns
        ---------
        color : (4,) uint8
          Average color of this material.
        """

    def add(self, material):
        """
        Adds new material to the container.

        Parameters
        ----------
        material : Material
            The material to be added.
        """
        self.materials.append(material)

    def get(self, idx):
        """
        Get material by index.

        Parameters
        ----------
        idx : int
            Index of the material to be retrieved.

        Returns
        -------
            The material on the given index.
        """
        return self.materials[idx]


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
                 baseColorFactor=None,
                 metallicFactor=None,
                 roughnessFactor=None,
                 normalTexture=None,
                 occlusionTexture=None,
                 baseColorTexture=None,
                 metallicRoughnessTexture=None,
                 doubleSided=False,
                 alphaMode=None,
                 alphaCutoff=None,
                 **kwargs):

        # store values in an internal dict
        self._data = {}

        # (3,) float
        self.emissiveFactor = emissiveFactor
        # (3,) or (4,) float with RGBA colors
        self.baseColorFactor = baseColorFactor

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

        if len(kwargs) > 0:
            util.log.debug(
                'unsupported material keys: {}'.format(
                    ', '.join(kwargs.keys())))

    @property
    def emissiveFactor(self):
        """
        The factors for the emissive color of the material.
        This value defines linear multipliers for the sampled
        texels of the emissive texture.

        Returns
        -----------
        emissiveFactor : (3,) float
           Ech element in the array MUST be greater than
           or equal to 0 and less than or equal to 1.
        """
        return self._data.get('emissiveFactor')

    @emissiveFactor.setter
    def emissiveFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('emissiveFactor', None)
        else:
            # non-None values must be a floating point
            emissive = np.array(value, dtype=np.float64).reshape(3)
            if emissive.min() < -_eps or emissive.max() > (1 + _eps):
                raise ValueError('all factors must be between 0.0-1.0')
            self._data['emissiveFactor'] = emissive

    @property
    def alphaMode(self):
        """
        The material alpha rendering mode enumeration
        specifying the interpretation of the alpha value of
        the base color.

        Returns
        -----------
        alphaMode : str
          One of 'OPAQUE', 'MASK', 'BLEND'
        """
        return self._data.get('alphaMode')

    @alphaMode.setter
    def alphaMode(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('alphaMode', None)
        else:
            # non-None values must be one of three values
            value = str(value).upper().strip()
            if value not in ['OPAQUE', 'MASK', 'BLEND']:
                raise ValueError('incorrect alphaMode: %s', value)
            self._data['alphaMode'] = value

    @property
    def alphaCutoff(self):
        """
        Specifies the cutoff threshold when in MASK alpha mode.
        If the alpha value is greater than or equal to this value
        then it is rendered as fully opaque, otherwise, it is rendered
        as fully transparent. A value greater than 1.0 will render
        the entire material as fully transparent. This value MUST be
        ignored for other alpha modes. When alphaMode is not defined,
        this value MUST NOT be defined.

        Returns
        -----------
        alphaCutoff : float
          Value of cutoff.
        """
        return self._data.get('alphaCutoff')

    @alphaCutoff.setter
    def alphaCutoff(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('alphaCutoff', None)
        else:
            self._data['alphaCutoff'] = float(value)

    @property
    def doubleSided(self):
        """
Specifies whether the material is double sided.

        Returns
        -----------
        doubleSided : bool
          Specifies whether the material is double sided.
        """
        return self._data.get('doubleSided')

    @doubleSided.setter
    def doubleSided(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('doubleSided', None)
        else:
            self._data['doubleSided'] = bool(value)

    @property
    def metallicFactor(self):
        """
        The factor for the metalness of the material. This value
        defines a linear multiplier for the sampled metalness values
        of the metallic-roughness texture.


        Returns
        -----------
        metallicFactor : float
          How metally is the material
        """
        return self._data.get('metallicFactor')

    @metallicFactor.setter
    def metallicFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('metallicFactor', None)
        else:
            self._data['metallicFactor'] = float(value)

    @property
    def roughnessFactor(self):
        """
        The factor for the roughness of the material. This value
        defines a linear multiplier for the sampled roughness values
        of the metallic-roughness texture.

        Returns
        -----------
        roughnessFactor : float
          Roughness of material.
        """
        return self._data.get('roughnessFactor')

    @roughnessFactor.setter
    def roughnessFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('roughnessFactor', None)
        else:
            self._data['roughnessFactor'] = float(value)

    @property
    def baseColorFactor(self):
        """
        The factors for the base color of the material. This
        value defines linear multipliers for the sampled texels
        of the base color texture.

        Returns
        ---------
        color : (4,) uint8
          RGBA color
        """
        return self._data.get('baseColorFactor')

    @baseColorFactor.setter
    def baseColorFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('baseColorFactor', None)
        else:
            # non-None values must be RGBA color
            self._data['baseColorFactor'] = color.to_rgba(value)

    @property
    def baseColorTexture(self):
        """
        The base color texture image.

        Returns
        ----------
        image : PIL.Image
          Color texture.
        """
        return self._data.get('baseColorTexture')

    @baseColorTexture.setter
    def baseColorTexture(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop('baseColorTexture', None)
        else:
            # non-None values must be RGBA color
            self._data['baseColorTexture'] = value

    def copy(self):
        # doing a straight deepcopy fails due to PIL images
        kwargs = {}
        # collect stored values as kwargs
        for k, v in self._data.items():
            if v is None:
                continue
            if hasattr(v, 'copy'):
                # use an objects explicit copy if available
                kwargs[k] = v.copy()
            else:
                # otherwise just hope deepcopy does something
                kwargs[k] = copy.deepcopy(v)
        return PBRMaterial(**kwargs)

    def to_color(self, uv):
        """
        Get the rough color at a list of specified UV
        coordinates.

        Parameters
        -------------
        uv : (n, 2) float
          UV coordinates on the material

        Returns
        -------------
        colors :
        """
        colors = color.uv_to_color(
            uv=uv, image=self.baseColorTexture)
        if colors is None and self.baseColorFactor is not None:
            colors = self.baseColorFactor.copy()
        return colors

    def to_simple(self):
        """
        Get a copy of the current PBR material as
        a simple material.

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
        return hash(bytes().join(
            np.asanyarray(v).tobytes()
            for v in self._data.values() if v is not None))


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
    try:
        from PIL import Image
    except BaseException as E:
        return exceptions.ExceptionModule(E)

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
        [unique_idx[hash(m)].append(i)
         for i, m in enumerate(materials)]
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
        xy_off = off / final_size
        # scale and translate each of the new UV coordinates
        # [new_uv.append((uvs[i] * scale) + uv_off) for i in idxs]
        # TODO : figure out why this is broken sometimes...

        def transform_uvs(uv, scale, xy_off):
            xy = np.stack([uv[:, 0], 1 - uv[:, 1]], axis=-1)
            xy = (xy * scale) + xy_off
            return np.stack([xy[:, 0], 1 - xy[:, 1]], axis=-1)

        [new_uv.append(transform_uvs(uvs[i], scale, xy_off)) for i in idxs]

    # stack UV coordinates into single (n, 2) array
    stacked = np.vstack(new_uv)

    return SimpleMaterial(image=final), stacked
