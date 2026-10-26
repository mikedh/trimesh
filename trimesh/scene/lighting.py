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
from ..typed import ArrayLike, NDArray, float64
from .cameras import current_view

# default light color
_DEFAULT_RGBA = np.array([60, 60, 60, 255], dtype=np.uint8)

# Rec.709 weights of linear RGB for how bright a color looks
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])

# the default studio's warm key and cool rim light colors, which glTF
# stores as linear RGB, and its slightly warm sky and floor: every part
# of the studio is as bright as white of the luminance it is given
KEY_COLOR = np.array([255, 219, 178])
RIM_COLOR = np.array([191, 212, 255])
SKY_COLOR = np.array([255, 242, 230])

# the default studio's (azimuth, elevation) in degrees of its key light,
# its rim light, and the softboxes on the key's side and the rim's side:
# azimuth is around up from towards the camera to its right, and the key
# is high so its shadow stays short and under the scene
STUDIO_ANGLES = np.radians([[45.0, 35.0], [210.0, 65.0], [60.0, 20.0], [-70.0, 12.0]])
# the same as unit vectors in the studio's frame: +X right, +Y up, +Z back
STUDIO_DIRECTIONS = np.column_stack(
    (
        np.sin(STUDIO_ANGLES[:, 0]) * np.cos(STUDIO_ANGLES[:, 1]),
        np.sin(STUDIO_ANGLES[:, 1]),
        np.cos(STUDIO_ANGLES[:, 0]) * np.cos(STUDIO_ANGLES[:, 1]),
    )
)
# tints scaled to a luminance so tinting changes color and not how
# bright it looks: a white surface facing a directional light of
# intensity `I` lux and linear color `C` reflects `C * I / pi`, which is
# 0.58 and 0.28 for the key and rim, then the sky and floor, and the
# softboxes overhead, on the key's side and on the rim's side
STUDIO_TINTS = np.array(
    [KEY_COLOR, RIM_COLOR, SKY_COLOR, SKY_COLOR, [255] * 3, KEY_COLOR, RIM_COLOR]
)
STUDIO_SCALE = np.array([0.58 * np.pi, 0.28 * np.pi, 0.31, 0.096, 0.35, 2.0, 1.0]) / (
    STUDIO_TINTS / 255.0 @ LUMINANCE
)
# the sharpness of the softboxes overhead and either side, which are
# spherical Gaussians as `studio_cubemap` describes
STUDIO_SHARPNESS = np.array([4.0, 30.0, 30.0])

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
# `[1, y, z, x, xy, yz, 3z^2 - 1, xz, x^2 - y^2]`, and a clamped cosine
# scales band l = 0, 1, 2 of radiance by pi, 2pi / 3 and pi / 4 into
# irradiance: Ramamoorthi and Hanrahan 2001, "An Efficient Representation
# for Irradiance Environment Maps", equations 3 and 8
HARMONICS = np.sqrt(np.array([1, 3, 3, 3, 15, 15, 5 / 4, 15, 15 / 4]) / (4.0 * np.pi))
IRRADIANCE_BANDS = np.pi * np.array([1, *[2 / 3] * 3, *[1 / 4] * 5])

# the fewest texels across the standard deviation `1 / sqrt(sharpness)`
# radians of a spherical Gaussian a cube map holds it in, where texels
# are `2 / n` radians across the middle of a face: a Gaussian's spectrum
# at the Nyquist frequency is then `exp(-2 pi^2)`, or 3e-9, of its peak
TEXELS_PER_DEVIATION = 2.0

# the default studio's cube map and irradiance, which only depend on
# the constants above so are made once
_STUDIO_ENVIRONMENT = {}


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
    Light arriving from infinitely far away from every direction, held as
    the draft glTF extension `KHR_lights_environment` describes it
    (https://github.com/KhronosGroup/glTF/pull/1956): a cube map of linear
    radiance in cd/m^2 and the l=2 spherical harmonics of the irradiance
    it gives, both in the frame of the node carrying it whose +Y is up,
    then multiplied by `color` and `intensity`. Only the OpenGL viewer
    draws it: glTF has no ratified extension for it so exporters skip it,
    and other engines light a GLB their own way.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA tint of the whole environment in linear space, white by default.
    intensity : float
        Multiplies the whole environment.
    cubemap : (6, n, n, 3) float
        Linear RGB radiance of every texel of the faces of `CUBE_FACES`,
        which can be passed as an equirectangular (h, 2h, 3) image.
    irradiance : (9, 3) float
        Light reaching a surface facing each direction from its
        hemisphere weighted by cosine, as coefficients of
        `spherical_harmonics`, computed from `cubemap` if not passed.
    """

    def __init__(
        self, name=None, color=None, intensity=None, cubemap=None, irradiance=None
    ):
        if color is None:
            # white so the cube map is the radiance
            color = [255, 255, 255, 255]
        super().__init__(name=name, color=color, intensity=intensity)
        if cubemap is None:
            # no light from anywhere
            cubemap = np.zeros((6, 1, 1, 3))
        cubemap = np.asarray(cubemap, dtype=np.float64)
        if cubemap.ndim == 3:
            cubemap = equirectangular_to_cubemap(cubemap)
        if irradiance is None:
            irradiance = cubemap_irradiance(cubemap)
        self.cubemap = cubemap
        self.irradiance = np.asarray(irradiance, dtype=np.float64).reshape((9, 3))

    def __hash__(self) -> int:
        """
        A hash of every value, which changes when any of them do.

        Returns
        ------------
        hashed : int
          Hash of the color, intensity, irradiance and cube map.
        """
        values = np.concatenate(
            (self.color / 255.0, [self.intensity], self.irradiance.ravel())
        )
        return caching.hash_fast(values.tobytes() + self.cubemap.tobytes())


def cubemap_directions(size: int) -> tuple[NDArray[float64], NDArray[float64]]:
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


def cubemap_size(sharpness: float) -> int:
    """
    The fewest texels across each face, as a power of two, which hold a
    spherical Gaussian `exp(sharpness * (cos(angle) - 1))` with
    `TEXELS_PER_DEVIATION` across its standard deviation of
    `1 / sqrt(sharpness)` radians, where texels in the middle of a face
    are `2 / size` radians wide.

    Parameters
    ------------
    sharpness : float
      Sharpness of the spherical Gaussian.

    Returns
    ------------
    size : int
      Texels across each face.
    """
    texels = 2.0 * TEXELS_PER_DEVIATION * np.sqrt(sharpness)
    return 2 ** int(np.ceil(np.log2(texels)))


def bilinear(images: NDArray, layers: NDArray, coordinates: NDArray) -> NDArray:
    """
    Sample images between their texels like OpenGL's linear filtering,
    clamping to their edges.

    Parameters
    ------------
    images : (k, h, w, c) float
      Layers of images.
    layers : (m,) int
      Which layer each sample is in.
    coordinates : (m, 2) float
      Row and column of each sample, where texel middles are integers.

    Returns
    ------------
    samples : (m, c) float
      Each sample blended from the four texels around it.
    """
    limit = np.array(images.shape[1:3]) - 1
    coordinates = np.clip(coordinates, 0, limit)
    low = np.floor(coordinates).astype(np.int64)
    fraction = (coordinates - low)[:, None]
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    index = np.minimum(low[:, None] + corners, limit)
    weights = np.prod(np.where(corners, fraction, 1.0 - fraction), axis=2)
    texels = images[layers[:, None], index[..., 0], index[..., 1]]
    return (texels * weights[..., None]).sum(axis=1)


def cubemap_sample(cubemap: ArrayLike, directions: ArrayLike) -> NDArray[float64]:
    """
    The radiance of a cube map towards directions, blending the four
    texels around each across the edges of faces like OpenGL's linear
    filtering with `GL_TEXTURE_CUBE_MAP_SEAMLESS`.

    Parameters
    ------------
    cubemap : (6, n, n, c) float
      Texels of the faces of `CUBE_FACES`.
    directions : (..., 3) float
      Directions to look in, of any length.

    Returns
    ------------
    radiance : (..., c) float
      Value of the cube map in each direction.
    """
    cubemap = np.asarray(cubemap, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    size = cubemap.shape[1]
    # the texels of every face with a border one texel past its edges
    middles = (2.0 * np.arange(-1, size + 1) + 1.0 - size) / size
    faces = CUBE_FACES[:, None, None]
    border = (
        faces[..., 0, :]
        + middles[:, None] * faces[..., 1, :]
        + middles[:, None, None] * faces[..., 2, :]
    )
    # the face each of those and the directions are on, then each in the
    # face's frame and as the row and column on it
    flat = np.vstack((border.reshape((-1, 3)), directions.reshape((-1, 3))))
    axis = np.abs(flat).argmax(axis=1)
    face = 2 * axis + (flat[np.arange(len(flat)), axis] < 0.0)
    local = np.matmul(CUBE_FACES[face], flat[:, :, None])[:, :, 0]
    coordinates = (local[:, [2, 1]] / local[:, :1] + 1.0) * size / 2.0 - 0.5
    # the border is the nearest texel on the neighboring face
    count = len(flat) - len(directions.reshape((-1, 3)))
    nearest = np.clip(np.round(coordinates[:count]), 0, size - 1).astype(np.int64)
    padded = cubemap[face[:count], nearest[:, 0], nearest[:, 1]]
    padded = padded.reshape((6, size + 2, size + 2, -1))
    samples = bilinear(padded, face[count:], coordinates[count:] + 1.0)
    return samples.reshape((*directions.shape[:-1], cubemap.shape[-1]))


def equirectangular_to_cubemap(image: ArrayLike, size: int | None = None) -> NDArray:
    """
    Resample an equirectangular image, a latitude-longitude panorama
    twice as wide as it is tall, into a cube map. Its top row is straight
    up +Y and a direction `d` is `0.5 + atan2(d[2], d[0]) / (2 pi)` of the
    way across it, as the Khronos glTF Sample Viewer reads them.

    Parameters
    ------------
    image : (h, 2h, c) float
      Linear radiance with the first row at the top.
    size : int or None
      Texels across each face, by default the smallest power of two whose
      texels in the middle of a face, `2 / size` radians, are no wider
      than the image's at its equator, `pi / h` radians.

    Returns
    ------------
    cubemap : (6, size, size, c) float
      Texels of the faces of `CUBE_FACES`.
    """
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape[:2]
    if size is None:
        size = 2 ** int(np.ceil(np.log2(max(2.0 * height / np.pi, 1.0))))
    x, y, z = cubemap_directions(size)[0].reshape((-1, 3)).T
    # a column from each edge on the other side so longitude wraps around
    wrapped = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    coordinates = np.column_stack(
        (
            np.arccos(np.clip(y, -1.0, 1.0)) / np.pi * height - 0.5,
            (0.5 + np.arctan2(z, x) / (2.0 * np.pi)) * width + 0.5,
        )
    )
    samples = bilinear(wrapped[None], np.zeros(len(x), dtype=np.int64), coordinates)
    return samples.reshape((6, size, size, -1))


def spherical_harmonics(directions: ArrayLike) -> NDArray[float64]:
    """
    The real spherical harmonics up to l=2 of unit vectors.

    Parameters
    ------------
    directions : (..., 3) float
      Unit vectors.

    Returns
    ------------
    harmonics : (..., 9) float
      Each harmonic in the order of `HARMONICS`.
    """
    x, y, z = np.moveaxis(np.asarray(directions, dtype=np.float64), -1, 0)
    linear = [np.ones_like(x), y, z, x]
    quadratic = [x * y, y * z, 3 * z**2 - 1, x * z, x**2 - y**2]
    return np.stack(linear + quadratic, axis=-1) * HARMONICS


def cubemap_irradiance(cubemap: ArrayLike) -> NDArray[float64]:
    """
    The irradiance of a cube map, which is light reaching a surface from
    its hemisphere weighted by cosine, as l=2 spherical harmonics: the
    harmonics closest to the radiance over the sphere, every texel
    weighted by its solid angle, which a clamped cosine scales by
    `IRRADIANCE_BANDS`. Radiance which is a sum of harmonics up to l=2,
    like a gradient, has its irradiance exactly.

    Parameters
    ------------
    cubemap : (6, n, n, c) float
      Radiance of the faces of `CUBE_FACES`.

    Returns
    ------------
    irradiance : (9, c) float
      Coefficients of `spherical_harmonics`.
    """
    cubemap = np.asarray(cubemap, dtype=np.float64)
    directions, solid_angle = cubemap_directions(cubemap.shape[1])
    weights = np.sqrt(solid_angle).reshape((-1, 1))
    radiance = np.linalg.lstsq(
        spherical_harmonics(directions).reshape((-1, 9)) * weights,
        cubemap.reshape((-1, cubemap.shape[-1])) * weights,
        rcond=None,
    )[0]
    return radiance * IRRADIANCE_BANDS[:, None]


def studio_cubemap(
    sky: ArrayLike, ground: ArrayLike, softboxes=(), size: int | None = None
) -> NDArray[float64]:
    """
    A cube map of a photo studio in the frame of its node whose +Y is up:
    a gradient linear in height from `ground` straight down to `sky`
    straight up, plus softboxes which are large soft lights infinitely
    far away, each a spherical Gaussian `color * exp(sharpness * (dot(d,
    direction) - 1))` in every direction `d`, where larger sharpness is a
    smaller and harder light.

    Parameters
    ------------
    sky : (3,) float
      Linear RGB radiance from straight up.
    ground : (3,) float
      Linear RGB radiance from straight down.
    softboxes : sequence of (direction, sharpness, color)
      Unit vector towards each softbox, its sharpness, and the linear RGB
      radiance at its middle.
    size : int or None
      Texels across each face, by default twice `cubemap_size` of the
      sharpest softbox so a mip level down still holds it, which a
      reflection blends in as glTF roughness is never exactly zero.

    Returns
    ------------
    cubemap : (6, size, size, 3) float
      Radiance of the faces of `CUBE_FACES`.
    """
    towards = util.unitize(np.reshape([box[0] for box in softboxes], (-1, 3)))
    sharpness = np.array([box[1] for box in softboxes], dtype=np.float64)
    colors = np.reshape([box[2] for box in softboxes], (-1, 3))
    if size is None:
        # a gradient is as smooth as a lobe of sharpness one
        size = 2 * cubemap_size(sharpness.max(initial=1.0))
    directions = cubemap_directions(size)[0]
    height = 0.5 + 0.5 * directions[..., 1:2]
    lobes = np.exp(sharpness * (directions @ towards.T - 1.0))
    ground = np.asarray(ground, dtype=np.float64)
    return ground + (np.asarray(sky, dtype=np.float64) - ground) * height + lobes @ colors


def studio_environment(
    scene, camera_transform: ArrayLike | None = None
) -> tuple[EnvironmentLight, NDArray[float64]]:
    """
    The default studio's environment, which the viewer draws when a
    scene has no `EnvironmentLight` of its own: a dim gradient from a
    dark floor to a pale sky, a broad light overhead and two softboxes
    either side of the camera which put window-like highlights on shiny
    surfaces, fixed in the world around where the camera is.

    Parameters
    --------------
    scene : trimesh.Scene
      Scene to light, which is not changed.
    camera_transform : None or (4, 4) float
      Where the camera is, defaulting to `current_view(scene)`.

    Returns
    --------------
    light : EnvironmentLight
      The studio, which the viewer draws and exporters skip.
    transform : (4, 4) float
      World pose of the studio, whose +Y is the world axis nearest the
      camera's up and whose +Z is level and back towards the camera.
    """
    if camera_transform is None:
        camera_transform = current_view(scene)[1]
    camera_transform = np.asarray(camera_transform, dtype=np.float64)
    # the world axis nearest to up on screen
    axes = np.vstack((np.eye(3), -np.eye(3)))
    up = axes[np.argmax(axes @ camera_transform[:3, 1])]
    # the camera's backwards direction made level, which is never
    # vertical as up is the axis nearest the camera's +Y, then the right
    # hand rule completes the studio's frame: +X right, +Y up and +Z back
    backwards = camera_transform[:3, 2]
    back = util.unitize(backwards - np.dot(backwards, up) * up)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((np.cross(up, back), up, back))
    try:
        cubemap, irradiance = _STUDIO_ENVIRONMENT["studio"]
    except KeyError:
        radiance = STUDIO_TINTS / 255.0 * STUDIO_SCALE[:, None]
        towards = [[0.0, 1.0, 0.0], *STUDIO_DIRECTIONS[2:]]
        cubemap = studio_cubemap(
            radiance[2], radiance[3], list(zip(towards, STUDIO_SHARPNESS, radiance[4:]))
        )
        irradiance = cubemap_irradiance(cubemap)
        # shared by every studio so nothing may change them
        cubemap.flags.writeable, irradiance.flags.writeable = False, False
        _STUDIO_ENVIRONMENT["studio"] = (cubemap, irradiance)
    name = util.unique_name("environment_light", set(scene.graph.nodes))
    light = EnvironmentLight(name=name, cubemap=cubemap, irradiance=irradiance)
    return light, frame


def autolight(
    scene, camera_transform: ArrayLike | None = None
) -> tuple[list[Light], NDArray[float64]]:
    """
    Generate lights for a scene that look decent: a photo studio fixed
    in the world around the camera, with a warm key light from in front,
    to the right and above and a cool rim light from high behind which
    outlines things. The viewer adds the environment of the studio from
    `studio_environment`.

    Parameters
    --------------
    scene : trimesh.Scene
      Scene to light, which is not changed.
    camera_transform : None or (4, 4) float
      Where the camera is, defaulting to `current_view(scene)`.

    Returns
    --------------
    lights : [Light]
      The key and rim `DirectionalLight`.
    transforms : (2, 4, 4) float
      World pose of each light.
    """
    frame = studio_environment(scene, camera_transform)[1]
    names = set(scene.graph.nodes)
    lights = [
        DirectionalLight(
            name=util.unique_name(name, names),
            color=[*color, 255],
            intensity=intensity,
        )
        for name, color, intensity in zip(
            ("key_light", "rim_light"), (KEY_COLOR, RIM_COLOR), STUDIO_SCALE[:2]
        )
    ]
    # a light's +Z points back towards where its light comes from
    transforms = [
        align_vectors([0.0, 0.0, 1.0], frame[:3, :3] @ towards)
        for towards in STUDIO_DIRECTIONS[:2]
    ]
    return lights, np.array(transforms)
