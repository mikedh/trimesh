from PIL import Image

from .interfaces import blender


def unwrap(mesh, image=None, **kwargs):
    result = blender.unwrap(mesh=mesh, **kwargs)

    if image is not None:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        result.visual.material.image = image

    return result
