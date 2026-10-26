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


# one large soft light infinitely far away in an `EnvironmentLight`, which
# is a spherical Gaussian `color * exp(sharpness * (dot(d, direction) - 1))`
# for every direction `d`: larger `sharpness` is a smaller, harder light
SOFTBOX = np.dtype(
    [
        # unit vector towards the softbox in the frame of the light's node
        ("direction", "<f8", 3),
        ("sharpness", "<f8"),
        # linear RGB radiance at its center
        ("color", "<f8", 3),
    ]
)


class EnvironmentLight(Light):
    """
    Light arriving from infinitely far away from every direction `d` in
    the frame of the node carrying it, whose +Y is up: `ground` from
    straight down to `sky` from straight up linear in `d[1]`, plus large
    soft lights which are rows of `SOFTBOX`, all multiplied by `color`
    and `intensity`. `EnvironmentLight(sky=c, ground=c)` is a constant
    ambient light `c`. glTF has no such light so it is stored in the
    `extras` of its node.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (4,) uint8
        RGBA tint of the whole environment in linear space, white by default.
    intensity : float
        Multiplies the whole environment.
    sky : (3,) float
        Linear RGB radiance from straight up.
    ground : (3,) float
        Linear RGB radiance from straight down.
    softboxes : (n,) SOFTBOX
        Each softbox's direction, sharpness and color.
    """

    def __init__(
        self,
        name=None,
        color=None,
        intensity=None,
        sky=None,
        ground=None,
        softboxes=None,
    ):
        if color is None:
            # white so `sky`, `ground` and the softboxes are the radiance
            color = [255, 255, 255, 255]
        super().__init__(name=name, color=color, intensity=intensity)
        if sky is None:
            sky = np.zeros(3)
        if ground is None:
            ground = np.zeros(3)
        if softboxes is None:
            softboxes = []
        self.sky = np.array(sky, dtype=np.float64).reshape(3)
        self.ground = np.array(ground, dtype=np.float64).reshape(3)
        # a `SOFTBOX` array, or anything it can be made from such as
        # `[(direction, sharpness, color), ...]`
        self.softboxes = np.array(softboxes, dtype=SOFTBOX).reshape(-1)

    def __hash__(self) -> int:
        """
        A hash of every value, which changes when any of them do.

        Returns
        ------------
        hashed : int
          Hash of the color, intensity, gradient and softboxes.
        """
        values = np.concatenate(
            (self.color / 255.0, [self.intensity], self.sky, self.ground)
        )
        return caching.hash_fast(values.tobytes() + self.softboxes.tobytes())


def autolight(
    scene, camera_transform: ArrayLike | None = None
) -> tuple[list[Light], NDArray[float64]]:
    """
    Generate lights for a scene that look decent: a photo studio fixed
    in the world around the camera, with a warm key light from in front,
    to the right and above, a cool rim light from high behind which
    outlines things, and an environment of a dim gradient from a dark
    floor to a pale sky, a broad light overhead and two softboxes either
    side of the camera which put window-like highlights on shiny surfaces.

    Parameters
    --------------
    scene : trimesh.Scene
      Scene to light, which is not changed.
    camera_transform : None or (4, 4) float
      Where the camera is, defaulting to `current_view(scene)`.

    Returns
    --------------
    lights : [Light]
      The key and rim `DirectionalLight` and an `EnvironmentLight`.
    transforms : (3, 4, 4) float
      World pose of each light.
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

    azimuth, elevation = STUDIO_ANGLES.T
    towards = np.column_stack(
        (
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
            np.cos(azimuth) * np.cos(elevation),
        )
    )
    # tints scaled to a luminance so tinting changes color and not how
    # bright it looks: a white surface facing a directional light of
    # intensity `I` lux and linear color `C` reflects `C * I / pi`, which is
    # 0.58 and 0.28 for the key and rim, then the sky and floor, and the
    # softboxes overhead, on the key's side and on the rim's side
    tints = np.array(
        [KEY_COLOR, RIM_COLOR, SKY_COLOR, SKY_COLOR, [255] * 3, KEY_COLOR, RIM_COLOR]
    )
    luminance = np.array([0.58 * np.pi, 0.28 * np.pi, 0.31, 0.096, 0.35, 2.0, 1.0])
    scale = luminance / (tints / 255.0 @ LUMINANCE)
    radiance = tints / 255.0 * scale[:, None]

    names = set(scene.graph.nodes)
    lights = [
        DirectionalLight(
            name=util.unique_name("key_light", names),
            color=[*KEY_COLOR, 255],
            intensity=scale[0],
        ),
        DirectionalLight(
            name=util.unique_name("rim_light", names),
            color=[*RIM_COLOR, 255],
            intensity=scale[1],
        ),
        EnvironmentLight(
            name=util.unique_name("environment_light", names),
            sky=radiance[2],
            ground=radiance[3],
            softboxes=list(
                zip([[0.0, 1.0, 0.0], *towards[2:]], [4.0, 30.0, 30.0], radiance[4:])
            ),
        ),
    ]
    # a light's +Z points back towards where its light comes from
    transforms = [align_vectors([0.0, 0.0, 1.0], frame[:3, :3] @ v) for v in towards[:2]]
    return lights, np.array([*transforms, frame])
