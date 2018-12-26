import numpy as np

from . import color
from .. import util


class TextureVisuals(object):
    def __init__(self,
                 vertex_uv=None,
                 textures=None):
        """
        Store vertex texture
        """

        self.vertex_uv = vertex_uv
        self.textures = textures

    def to_color(self):
        viz = color.ColorVisuals(vertex_colors=uv_to_color(
            self.vertex_uv, next(iter(self.textures.values()))))
        return viz

    def update_vertices(self, mask):
        """
        Apply a mask to remove or duplicate vertex properties.
        """
        self.vertex_uv = self.vertex_uv[mask]

    def update_faces(self, mask):
        """
        Apply a mask to remove or duplicate face properties
        """
        pass


class Material(object):
    pass


class PBRMaterial(Material):
    """
    Create a material for physically based rendering.
    """
    def __init__(self,
                 name=None,
                 emissiveFactor=None,
                 emissiveTexture=None,
                 normalTexture=None,
                 occlusionTexture=None,
                 pbrBaseColorTexture=None,
                 pbrMetallicRoughnessTexture=None):
        
        self.emissiveFactor=emissiveFactor
        self.emissiveTexture=emissiveTexture
        self.normalTexture=normalTexture
        self.occlusionTexture=occlusionTexture
        self.pbrBaseColorTexture=pbrBaseColorTexture
        self.pbrMetallicRoughnessTexture=pbrMetallicRoughnessTexture

def load(names, resolver):
    """
    Load named textures using a resolver into a PIL image.

    Parameters
    --------------
    names : list of str
      Name of texture files
    resolver : Resolver
      Object to get raw data of texture file

    Returns
    ---------------
    textures : dict
      name : PIL.Image
    """
    # import here for soft dependency
    import PIL
    textures = {}
    for name in names:
        data = resolver.get(name)
        image = PIL.Image.open(util.wrap_as_stream(data))
        textures[name] = image
    return textures


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
    uv = np.asanyarray(uv, dtype=np.float64)

    # get texture image pixel positions of UV coordinates
    x = (uv[:, 0] * (image.width - 1)).round().astype(np.int64)
    y = ((1 - uv[:, 1]) * (image.height - 1)).round().astype(np.int64)

    # wrap to image size in the manner of GL_REPEAT
    x %= image.width
    y %= image.height

    # access colors from pixel locations
    colors = np.asarray(image)[y, x]

    # handle greyscale
    if colors.ndim == 1:
        colors = np.repeat(colors[:, None], 3, axis=1)
    # now ndim == 2
    if colors.shape[1] == 3:
        colors = color.to_rgba(colors)  # rgb -> rgba
    return colors
