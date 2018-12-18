import numpy as np

from .color import to_rgba


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
    # https://shapenet.org/qaforum/index.php?qa=15&qa_1=why-is-the-texture-coordinate-in-the-obj-file-not-in-the-range  # NOQA
    if not (uv.min() >= 0 and uv.max() <= 1):
        uv = uv - np.floor(uv)

    x = (uv[:, 0] * (image.width - 1)).round().astype(int)
    y = ((1 - uv[:, 1]) * (image.height - 1)).round().astype(int)

    image = np.asarray(image)
    colors = image[y, x]
    if colors.ndim == 1:  # gray scale
        colors = np.repeat(colors[:, None], 3, axis=1)
    # now ndim == 2
    if colors.shape[1] == 3:
        colors = to_rgba(colors)  # rgb -> rgba
    return colors
