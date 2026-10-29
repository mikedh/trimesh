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

# cube map faces +X, -X, +Y, -Y, +Z, -Z as (normal, row axis, column axis)
# per table 8.19 of the OpenGL 4.6 core specification
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

# real spherical harmonics to l=2 are these times
# `[1, y, z, x, xy, yz, 3z^2 - 1, xz, x^2 - y^2]`, Ramamoorthi 2001 eq. 3
HARMONICS = np.sqrt(np.array([1, 3, 3, 3, 15, 15, 5 / 4, 15, 15 / 4]) / (4.0 * np.pi))
HARMONICS *= [1, -1, 1, -1, 1, -1, 1, -1, 1]

# spherical Gaussian sharpness matching a GGX lobe, fit within 1.1%:
# `exp(polyval(SPECULAR_SHARPNESS, log(roughness)))`
SPECULAR_SHARPNESS = (0.131, 1.0381, -0.7256, 1.0728)

# results of `prefilter` and `spherical_harmonics` by content
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
            self._color = _DEFAULT_RGBA.copy()
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
    Image based light from a cube map, as in `EXT_lights_image_based`.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    intensity : float
        Scale of the radiance.
    cubemap : (6, n, n, 3) float
        Linear RGB radiance, or an (h, 2h, 3) equirectangular image.
    """

    def __init__(
        self,
        name: str | None = None,
        intensity: Floating | None = None,
        cubemap: ArrayLike | None = None,
    ):
        super().__init__(name=name, color=[255, 255, 255, 255], intensity=intensity)
        if cubemap is None:
            cubemap = np.zeros((6, 1, 1, 3))
        cubemap = np.asarray(cubemap, dtype=np.float64)
        if cubemap.ndim == 3:
            # equirectangular with +Y up, as the Khronos Sample Viewer reads
            # it, resampled to power of two faces no coarser than its equator
            height, width = cubemap.shape[:2]
            size = 2 ** int(np.ceil(np.log2(max(2.0 * height / np.pi, 1.0))))
            x, y, z = cubemap_directions(size)[0].reshape((-1, 3)).T
            # pad columns so longitude wraps
            wrapped = np.concatenate((cubemap[:, -1:], cubemap, cubemap[:, :1]), axis=1)
            # bilinear blend of the four nearest pixels
            limit = np.array(wrapped.shape[:2]) - 1
            row = np.arccos(np.clip(y, -1.0, 1.0)) / np.pi * height - 0.5
            column = (0.5 + np.arctan2(z, x) / (2.0 * np.pi)) * width + 0.5
            coordinates = np.column_stack((row, column)).clip(0, limit)
            low = np.floor(coordinates).astype(np.int64)
            fraction = (coordinates - low)[:, None]
            corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            index = np.minimum(low[:, None] + corners, limit)
            weights = np.prod(np.where(corners, fraction, 1.0 - fraction), axis=2)
            samples = wrapped[index[..., 0], index[..., 1]] * weights[..., None]
            cubemap = samples.sum(axis=1).reshape((6, size, size, -1))
        self.cubemap = cubemap


def cubemap_directions(size: Integer) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    Direction and solid angle of every texel of a cube map.

    Parameters
    ------------
    size : int
      Texels across each face.

    Returns
    ------------
    directions : (6, size, size, 3) float
      Unit vector through each texel.
    solid_angle : (6, size, size) float
      Steradians of each texel.
    """
    edges = np.linspace(-1.0, 1.0, size + 1)
    middles = (edges[1:] + edges[:-1]) / 2.0
    texels = np.stack(np.broadcast_arrays(1.0, middles, middles[:, None]), axis=-1)
    directions = texels @ CUBE_FACES[:, None]
    # solid angle from face center to corner (u, v), differenced into texels
    corner = np.arctan2(
        np.outer(edges, edges), np.sqrt(np.add.outer(edges**2, edges**2) + 1)
    )
    solid_angle = corner[1:, 1:] - corner[:-1, 1:] - corner[1:, :-1] + corner[:-1, :-1]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions, np.broadcast_to(solid_angle, (6, size, size))


def prefilter(cubemap: ArrayLike) -> list[NDArray]:
    """
    Specular mip levels of a cube map for `EXT_lights_image_based`.

    Parameters
    ------------
    cubemap : (6, n, n, 3) float
      Radiance, `n` a power of two.

    Returns
    ------------
    levels : list of (6, m, m, 3) float32
      Mip levels, level `k` blurred to roughness `k / (n - 1)`.
    """
    cubemap = np.asarray(cubemap, dtype=np.float64)
    key = ("prefilter", caching.hash_fast(cubemap.tobytes()), cubemap.shape)
    try:
        return _ENVIRONMENT_DERIVED[key]
    except KeyError:
        pass
    # blurring is a dense texel-by-texel matrix, which bounds the mirror size
    mirror = 64
    roughest = 2 ** int(np.ceil(np.log2(4.0 * np.exp(SPECULAR_SHARPNESS[-1] / 2.0))))
    count = 1 + int(np.log2(mirror // roughest))
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
            # BLAS is much faster with a contiguous transpose
            cosine = directions @ np.ascontiguousarray(directions.T)
            # in place to stay float32
            weights = np.exp(sharpness * (cosine - 1.0))
            weights *= solid_angle.reshape(-1)
            blurred = (weights @ blurred.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        levels.append(blurred.reshape((6, width, width, 3)))
    _ENVIRONMENT_DERIVED[key] = levels
    return levels


def spherical_harmonics(cubemap: ArrayLike) -> NDArray[float64]:
    """
    Irradiance coefficients of a cube map, Ramamoorthi 2001 eq. 8.

    Parameters
    ------------
    cubemap : (6, n, n, c) float
      Radiance.

    Returns
    ------------
    irradiance : (9, c) float
      Coefficient of each harmonic per channel.
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
    Generate studio lighting for a scene: key, rim and environment.

    Parameters
    --------------
    scene : trimesh.Scene
      Scene to light, not modified.
    camera_transform : None or (4, 4) float
      Camera pose, default `current_view(scene)`.

    Returns
    --------------
    lights : [Light]
      Key and rim `DirectionalLight` and an `EnvironmentLight`.
    transforms : (3, 4, 4) float
      World pose of each light.
    """
    if camera_transform is None:
        camera_transform = current_view(scene)[1]
    camera_transform = np.asarray(camera_transform, dtype=np.float64)
    # +Y is the world axis nearest screen up, +Z the leveled camera backwards
    axes = np.vstack((np.eye(3), -np.eye(3)))
    up = axes[np.argmax(axes @ camera_transform[:3, 1])]
    backwards = camera_transform[:3, 2]
    back = util.unitize(backwards - np.dot(backwards, up) * up)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((np.cross(up, back), up, back))

    # key, rim and three softboxes as (elevation, azimuth from +Z to +X)
    elevation, azimuth = np.radians(
        [[35, 45], [65, -150], [90, 0], [20, 60], [12, -70]]
    ).T
    towards = np.column_stack(
        (
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.cos(azimuth),
        )
    )
    names = set(scene.graph.nodes)
    # (name, linear RGB, intensity in lux)
    studio = (
        ("key_light", (255, 219, 178), 2.077136),
        ("rim_light", (191, 212, 255), 1.064896),
    )
    lights = [
        DirectionalLight(
            name=util.unique_name(name, names), color=[*color, 255], intensity=intensity
        )
        for name, color, intensity in studio
    ]
    # a light's +Z points towards its source
    transforms = [align_vectors([0, 0, 1], frame[:3, :3] @ row) for row in towards[:2]]

    # floor-to-sky gradient plus spherical Gaussian softboxes, linear RGB
    directions = cubemap_directions(64)[0]
    lobes = np.exp([4.0, 30.0, 30.0] * (directions @ towards[2:].T - 1.0))
    rise = 0.5 + 0.5 * directions[..., 1:2]
    cubemap = np.concatenate((1.0 - rise, rise, lobes), axis=-1) @ [
        [0.10037, 0.095253, 0.09053],
        [0.324112, 0.307588, 0.292336],
        [0.35, 0.35, 0.35],
        [2.279906, 1.958037, 1.591464],
        [0.90676, 1.006457, 1.210596],
    ]
    environment = EnvironmentLight(
        name=util.unique_name("environment_light", names), cubemap=cubemap
    )
    return [*lights, environment], np.array([*transforms, frame])
