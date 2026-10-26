"""
lighting.py
--------------

Hold basic information about lights.

Forked from the light model in `pyrender`:
https://github.com/mmatl/pyrender
"""

import numpy as np

from .. import caching, util, visual
from ..geometry import align_vectors
from ..typed import ArrayLike, Floating, Integer, NDArray, float64
from .cameras import current_view

# default light color
_DEFAULT_RGBA = np.array([60, 60, 60, 255], dtype=np.uint8)

# The faces of a cube map in the order KTX2 and OpenGL store them, +X, -X,
# +Y, -Y, +Z, -Z, each as (towards it, along its rows, down its columns):
# the texel in row `t` and column `s` of a face `n` texels across faces
# `face[0] + (2 * s + 1 - n) / n * face[1] + (2 * t + 1 - n) / n * face[2]`
# as table 8.19 of the OpenGL 4.6 core specification lays them out
CUBE_FACES = np.array(
    [
        [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
        [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
        [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
        [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
    ],
    dtype=np.float64,
)

# the real spherical harmonics up to l=2 are these constants times
# `[1, y, z, x, xy, yz, 3z^2 - 1, xz, x^2 - y^2]`, with the Condon-Shortley
# phase as Babylon.js reads `EXT_lights_image_based`: Ramamoorthi and
# Hanrahan 2001, "An Efficient Representation for Irradiance Environment
# Maps", equation 3
HARMONICS = np.sqrt(np.array([1, 3, 3, 3, 15, 15, 5 / 4, 15, 15 / 4]) / (4.0 * np.pi))
HARMONICS *= [1, -1, 1, -1, 1, -1, 1, -1, 1]

# the spherical Gaussian with the same mean cosine as a GGX reflection
# lobe, whose sharpness is `exp(polyval(SPECULAR_SHARPNESS, log(roughness)))`
# fit within 1.1% over roughness 0.04 to 1, which blurs the environment
SPECULAR_SHARPNESS = (0.131, 1.0381, -0.7256, 1.0728)

# `prefilter` and `spherical_harmonics` by function and cube map content
_ENVIRONMENT_DERIVED = {}


class Light(util.ABC):
    """
    Base class for all light objects.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light: point and spot lights use luminous intensity
        in candela (lm/sr) while directional lights use illuminance
        in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights
        Must be > 0.0
        If None, the radius is assumed to be infinite.
    """

    def __init__(self, name=None, color=None, intensity=None, radius=None):
        if name is None:
            # if name is not passed, make it something unique
            self.name = f"light_{util.unique_id(6).upper()}"
        else:
            # otherwise assign it
            self.name = name

        self.color = color
        self.intensity = intensity
        self.radius = radius

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if value is None:
            self._color = _DEFAULT_RGBA
        else:
            value = visual.to_rgba(value)
            if len(value.shape) == 2:
                value = value[0]
            if value.shape != (4,):
                raise ValueError("couldn't convert color to RGBA!")
            # uint8 RGB color
            self._color = value

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        if value is not None:
            self._intensity = float(value)
        else:
            self._intensity = 1.0

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value is None or value < 0.0:
            self._radius = value
        else:
            self._radius = float(value)


class DirectionalLight(Light):
    """
    Directional lights are light sources that act as though they are
    infinitely far away and emit light in the direction of the local -z axis.
    This light type inherits the orientation of the node that it belongs to;
    position and scale are ignored except for their effect on the inherited
    node orientation. Because it is at an infinite distance, the light is
    not attenuated. Its intensity is defined in lumens per metre squared,
    or lux (lm/m2).

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) unit8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the radius is assumed to be infinite.
    """

    def __init__(self, name=None, color=None, intensity=None, radius=None):
        super().__init__(name=name, color=color, intensity=intensity, radius=radius)


class PointLight(Light):
    """
    Point lights emit light in all directions from their position in space;
    rotation and scale are ignored except for their effect on the inherited
    node position. The brightness of the light attenuates in a physically
    correct manner as distance increases from the light's position (i.e.
    brightness goes like the inverse square of the distance). Point light
    intensity is defined in candela, which is lumens per square radian (lm/sr).

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the radius is assumed to be infinite.
    """

    def __init__(self, name=None, color=None, intensity=None, radius=None):
        super().__init__(name=name, color=color, intensity=intensity, radius=radius)


class SpotLight(Light):
    """
    Spot lights emit light in a cone in the direction of the local -z axis.
    The angle and falloff of the cone is defined using two numbers, the
    `innerConeAngle` and `outerConeAngle`. As with point lights, the brightness
    also attenuates in a physically correct manner as distance increases from
    the light's position (i.e. brightness goes like the inverse square of the
    distance). Spot light intensity refers to the brightness inside the
    `innerConeAngle` (and at the location of the light) and is defined in
    candela, which is lumens per square radian (lm/sr). A spot light's position
    and orientation are inherited from its node transform. Inherited scale does
    not affect cone shape, and is ignored except for its effect on position
    and orientation.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend
        on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    radius : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the radius is assumed to be infinite.
    innerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff begins.
        Must be greater than or equal to `0` and less than `outerConeAngle`.
    outerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff ends.
        Must be greater than `innerConeAngle` and less than or equal to `PI / 2.0`.
    """

    def __init__(
        self,
        name=None,
        color=None,
        intensity=None,
        radius=None,
        innerConeAngle=0.0,
        outerConeAngle=np.pi / 4.0,
    ):
        super().__init__(name=name, color=color, intensity=intensity, radius=radius)
        self.outerConeAngle = outerConeAngle
        self.innerConeAngle = innerConeAngle

    @property
    def innerConeAngle(self):
        return self._innerConeAngle

    @innerConeAngle.setter
    def innerConeAngle(self, value):
        if value < 0.0 or value > self.outerConeAngle:
            raise ValueError("Invalid value for inner cone angle")
        self._innerConeAngle = float(value)

    @property
    def outerConeAngle(self):
        return self._outerConeAngle

    @outerConeAngle.setter
    def outerConeAngle(self, value):
        if value < 0.0 or value > np.pi / 2.0 + 1e-9:
            raise ValueError("Invalid value for outer cone angle")
        self._outerConeAngle = float(value)


class EnvironmentLight(Light):
    """
    Light from infinitely far away in every direction: a cube map of
    linear radiance in the frame of the node carrying it, times
    `intensity`. This is the image based light of the glTF extension
    `EXT_lights_image_based`, whose mip levels and coefficients are
    `prefilter` and `spherical_harmonics` of the cube map.

    Attributes
    ----------
    name : str, optional
        Name of the light and of its node.
    intensity : float
        Multiplies the whole environment.
    cubemap : (6, n, n, 3) float
        Linear RGB radiance of the faces of `CUBE_FACES`, or an
        equirectangular (h, 2h, 3) image to convert.
    """

    def __init__(
        self,
        name: str | None = None,
        intensity: Floating | None = None,
        cubemap: ArrayLike | None = None,
    ):
        # white as the cube map is the color, and dark unless passed one
        super().__init__(name=name, color=[255, 255, 255, 255], intensity=intensity)
        if cubemap is None:
            cubemap = np.zeros((6, 1, 1, 3))
        cubemap = np.asarray(cubemap, dtype=np.float64)
        if cubemap.ndim == 3:
            cubemap = equirectangular_to_cubemap(cubemap)
        self.cubemap = cubemap


def cubemap_directions(size: Integer) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    The direction through the middle of every texel of a cube map, and
    the exact solid angle each texel covers.

    Parameters
    ------------
    size : int
      Texels across each face.

    Returns
    ------------
    directions : (6, size, size, 3) float
      Unit vector through each texel by face, row and column.
    solid_angle : (6, size, size) float
      Steradians each texel covers, which sum to `4 pi`.
    """
    size = int(size)
    edges = np.linspace(-1.0, 1.0, size + 1)
    middles = (edges[1:] + edges[:-1]) / 2.0
    faces = CUBE_FACES[:, None, None]
    directions = (
        faces[..., 0, :]
        + middles[:, None] * faces[..., 1, :]
        + middles[:, None, None] * faces[..., 2, :]
    )
    # the solid angle from the middle of a face to the corner (u, v) is
    # `atan2(u * v, sqrt(u^2 + v^2 + 1))`, which differences into texels
    corner = np.arctan2(
        np.outer(edges, edges), np.sqrt(np.add.outer(edges**2, edges**2) + 1)
    )
    solid_angle = corner[1:, 1:] - corner[:-1, 1:] - corner[1:, :-1] + corner[:-1, :-1]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions, np.broadcast_to(solid_angle, (6, size, size))


def equirectangular_to_cubemap(
    image: ArrayLike, size: Integer | None = None
) -> NDArray[float64]:
    """
    Resample a latitude-longitude panorama twice as wide as it is tall
    into a cube map, blending the four pixels around each texel. Its top
    row is +Y and a direction `d` is `0.5 + atan2(d[2], d[0]) / (2 pi)`
    of the way across, as the Khronos glTF Sample Viewer reads them.

    Parameters
    ------------
    image : (h, 2h, c) float
      Linear radiance with the first row at the top.
    size : int or None
      Texels across each face, by default the smallest power of two whose
      texels mid-face, `2 / size` radians, are no wider than the image's
      at its equator, `pi / h` radians.

    Returns
    ------------
    cubemap : (6, size, size, c) float
      Texels of the faces of `CUBE_FACES`.
    """
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape[:2]
    if size is None:
        size = 2 ** int(np.ceil(np.log2(max(2.0 * height / np.pi, 1.0))))
    size = int(size)
    x, y, z = cubemap_directions(size)[0].reshape((-1, 3)).T
    # a column from each edge on the other side so longitude wraps around
    wrapped = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    # the row and column of each texel where pixel middles are integers
    limit = np.array(wrapped.shape[:2]) - 1
    row = np.arccos(np.clip(y, -1.0, 1.0)) / np.pi * height - 0.5
    column = (0.5 + np.arctan2(z, x) / (2.0 * np.pi)) * width + 0.5
    coordinates = np.column_stack((row, column)).clip(0, limit)
    low = np.floor(coordinates).astype(np.int64)
    fraction = (coordinates - low)[:, None]
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    index = np.minimum(low[:, None] + corners, limit)
    weights = np.prod(np.where(corners, fraction, 1.0 - fraction), axis=2)
    samples = (wrapped[index[..., 0], index[..., 1]] * weights[..., None]).sum(axis=1)
    return samples.reshape((6, size, size, -1))


def prefilter(cubemap: ArrayLike) -> list[NDArray]:
    """
    The mip levels of a cube map which reflections sample, the specular
    images of `EXT_lights_image_based`: level `k` of `n` is the radiance
    blurred by the spherical Gaussian as wide as a GGX lobe of roughness
    `k / (n - 1)`, integrated over every texel of the level. Cached by
    content.

    Parameters
    ------------
    cubemap : (6, n, n, 3) float
      Radiance of the faces of `CUBE_FACES`, `n` a power of two.

    Returns
    ------------
    levels : list of (6, m, m, 3) float32
      Every mip level, each half as wide as the one before.
    """
    cubemap = np.asarray(cubemap, dtype=np.float64)
    key = ("prefilter", caching.hash_fast(cubemap.tobytes()), cubemap.shape)
    try:
        return _ENVIRONMENT_DERIVED[key]
    except KeyError:
        pass
    # each blurred level is a dense matrix between all of its texels, 150 MB
    # of float32 at 32 across, so the mirror level is at most twice that,
    # and the roughest level holds the lobe of roughness one with two texels
    # across its deviation `1 / sqrt(exp(SPECULAR_SHARPNESS[-1]))`
    mirror = 64
    roughest = 2 ** int(np.ceil(np.log2(4.0 * np.exp(SPECULAR_SHARPNESS[-1] / 2.0))))
    count = 1 + int(np.log2(mirror // roughest))
    # a smaller cube map repeats each texel up to the mirror level, and
    # float32 is twice as fast and rounds `d @ d.T - 1` under 1e-7
    repeat = max(mirror // cubemap.shape[1], 1)
    cubemap = cubemap.repeat(repeat, axis=1).repeat(repeat, axis=2).astype(np.float32)
    levels = []
    for level in range(count):
        width = mirror >> level
        step = cubemap.shape[1] // width
        blurred = cubemap.reshape((6, width, step, width, step, 3)).mean(axis=(2, 4))
        if level > 0:
            log_roughness = np.log(level / (count - 1))
            sharpness = np.float32(np.exp(np.polyval(SPECULAR_SHARPNESS, log_roughness)))
            directions, solid_angle = cubemap_directions(width)
            directions = directions.reshape((-1, 3)).astype(np.float32)
            # BLAS multiplies by a contiguous transpose five times faster
            cosine = directions @ np.ascontiguousarray(directions.T)
            weights = np.exp(sharpness * (cosine - 1.0))
            weights *= solid_angle.reshape(-1)
            blurred = (weights @ blurred.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        levels.append(blurred.reshape((6, width, width, 3)))
    _ENVIRONMENT_DERIVED[key] = levels
    return levels


def spherical_harmonics(cubemap: ArrayLike) -> NDArray[float64]:
    """
    The irradiance of a cube map, the `irradianceCoefficients` of
    `EXT_lights_image_based`: the coefficients of `HARMONICS` closest to
    its radiance over the sphere, every texel weighted by its solid angle,
    with band l = 0, 1, 2 scaled by pi, 2 pi / 3 and pi / 4 as a clamped
    cosine does (Ramamoorthi and Hanrahan 2001, equation 8, and Babylon.js
    `SHCosKernelConvolution`). Radiance which is harmonics up to l=2, like
    a gradient, has its irradiance exactly. Cached by content.

    Parameters
    ------------
    cubemap : (6, n, n, c) float
      Radiance of the faces of `CUBE_FACES`.

    Returns
    ------------
    irradiance : (9, c) float
      Coefficient of each harmonic for each channel.
    """
    cubemap = np.asarray(cubemap, dtype=np.float64)
    key = ("harmonics", caching.hash_fast(cubemap.tobytes()), cubemap.shape)
    try:
        return _ENVIRONMENT_DERIVED[key]
    except KeyError:
        pass
    directions, solid_angle = cubemap_directions(cubemap.shape[1])
    x, y, z = directions.reshape((-1, 3)).T
    weights = np.sqrt(solid_angle).reshape((-1, 1))
    basis = (np.ones_like(x), y, z, x, x * y, y * z, 3 * z**2 - 1, x * z, x**2 - y**2)
    radiance = np.linalg.lstsq(
        np.column_stack(basis) * HARMONICS * weights,
        cubemap.reshape((-1, cubemap.shape[-1])) * weights,
        rcond=None,
    )[0]
    bands = np.pi * np.array([1, *[2 / 3] * 3, *[1 / 4] * 5])
    _ENVIRONMENT_DERIVED[key] = radiance * bands[:, None]
    return _ENVIRONMENT_DERIVED[key]


def autolight(
    scene, camera_transform: ArrayLike | None = None
) -> tuple[list[Light], NDArray[float64]]:
    """
    Generate lights for a scene that look decent: a photo studio fixed
    in the world around the camera, with a warm key light from in front,
    to the right and above, a cool rim light from high behind which
    outlines things, and an environment of a warm sky and floor with a
    broad softbox overhead and smaller ones on the key's and rim's side.

    Parameters
    --------------
    scene : trimesh.Scene
      Scene to light, which is not changed.
    camera_transform : None or (4, 4) float
      Where the camera is, defaulting to `current_view(scene)`.

    Returns
    --------------
    lights : [Light]
      The key and rim `DirectionalLight` and the `EnvironmentLight`.
    transforms : (3, 4, 4) float
      World pose of each light.
    """
    if camera_transform is None:
        camera_transform = current_view(scene)[1]
    camera_transform = np.asarray(camera_transform, dtype=np.float64)
    # the studio stands with +Y the world axis nearest up on screen and +Z
    # the camera's backwards made level, which is never vertical as up is
    # the axis nearest the camera's +Y, and +X completing a right hand
    axes = np.vstack((np.eye(3), -np.eye(3)))
    up = axes[np.argmax(axes @ camera_transform[:3, 1])]
    backwards = camera_transform[:3, 2]
    back = util.unitize(backwards - np.dot(backwards, up) * up)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((np.cross(up, back), up, back))

    names = set(scene.graph.nodes)
    lights, transforms = [], []
    # (name, linear RGB, intensity in lux, towards it in the studio frame)
    # where a white surface facing the key reflects 0.58 and the rim 0.28
    for name, color, intensity, towards in (
        ("key_light", (255, 219, 178), 2.077136, (0.579228, 0.573576, 0.579228)),
        ("rim_light", (191, 212, 255), 1.064896, (-0.211309, 0.906308, -0.365998)),
    ):
        lights.append(
            DirectionalLight(
                name=util.unique_name(name, names),
                color=[*color, 255],
                intensity=intensity,
            )
        )
        # a light's +Z points back towards where its light comes from
        transforms.append(align_vectors([0.0, 0.0, 1.0], frame[:3, :3] @ towards))

    # the environment, two texels across the deviation `1 / sqrt(30)` of
    # its side softboxes: a gradient from the floor straight down to the sky
    # straight up, in linear RGB
    directions = cubemap_directions(64)[0]
    ground = np.array([0.10037, 0.095253, 0.09053])
    sky = np.array([0.324112, 0.307588, 0.292336])
    cubemap = ground + np.multiply.outer(0.5 + 0.5 * directions[..., 1], sky - ground)
    # and softboxes `color * exp(sharpness * (dot(d, towards) - 1))` in every
    # direction `d` as (towards, sharpness, color): broad overhead, then on
    # the key's side and the rim's side
    for towards, sharpness, color in (
        ((0.0, 1.0, 0.0), 4.0, (0.35, 0.35, 0.35)),
        ((0.813798, 0.34202, 0.469846), 30.0, (2.279906, 1.958037, 1.591464)),
        ((-0.919158, 0.207912, 0.334546), 30.0, (0.90676, 1.006457, 1.210596)),
    ):
        lobe = np.exp(sharpness * (directions @ util.unitize(towards) - 1.0))
        cubemap += np.multiply.outer(lobe, color)
    lights.append(
        EnvironmentLight(
            name=util.unique_name("environment_light", names), cubemap=cubemap
        )
    )
    transforms.append(frame)
    return lights, np.array(transforms)
