"""
metadata.py
-------------

How a scene is lit beyond its own lights, which glTF has no place for.

`RenderSettings` lives in `scene.metadata["render"]` only when it was
changed from the defaults, so it round trips through glTF scene extras
without every file growing a copy of the defaults. When a scene is
rendered `RenderSettings.resolve` fills in every default for the
camera being rendered from, and the result is never stored.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from .. import util
from ..bounds import corners
from ..geometry import align_vectors
from ..typed import ArrayLike, NDArray, Number, float64
from .lighting import DirectionalLight

# the radius ambient occlusion looks for nearby surfaces within, as a
# fraction of `scene.scale` which is the length of the bounds diagonal
OCCLUSION_RADIUS = 0.1

# the sharpness of a spherical Gaussian whose mean cosine is 2/3, the
# same as the clamped cosine lobe which diffuse surfaces reflect with
DIFFUSE_SHARPNESS = 2.95

# the spherical Gaussian with the same mean cosine as a GGX reflection
# lobe, whose sharpness is `exp(polyval(SPECULAR_SHARPNESS, log(roughness)))`
# fit within 1.1% over roughness 0.04 to 1 where 1 is `DIFFUSE_SHARPNESS`
SPECULAR_SHARPNESS = (0.131, 1.0381, -0.7256, 1.0728)

# Rec.709 weights of linear RGB for how bright a color looks
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])

# the default studio's warm key and cool rim light colors, which glTF
# stores as linear RGB, and its slightly warm sky and floor: every part
# of the studio is as bright as white of the luminance it is given
KEY_COLOR = np.array([255, 219, 178])
RIM_COLOR = np.array([191, 212, 255])
SKY_COLOR = np.array([255, 242, 230])


@dataclass
class Softbox:
    """
    A large soft light infinitely far away, which is a spherical
    Gaussian: `color * exp(sharpness * (dot(d, direction) - 1))` for
    every direction `d`.
    """

    # unit vector from the scene towards the softbox
    direction: ArrayLike
    # larger is a smaller and harder light
    sharpness: Number
    # linear RGB radiance at its center
    color: ArrayLike

    def __post_init__(self):
        """
        Store plain floats so settings compare equal after a round
        trip through JSON whatever array type they were made from.
        """
        self.direction = [float(value) for value in np.reshape(self.direction, 3)]
        self.sharpness = float(self.sharpness)
        self.color = [float(value) for value in np.reshape(self.color, 3)]


@dataclass
class Environment:
    """
    Light arriving from infinitely far away from every direction `d`:
    `ground` from straight down to `sky` from straight up, linear in
    `dot(d, up)`, plus softboxes. `Environment(sky=c, ground=c)` is a
    constant ambient light `c`.
    """

    # unit vector which is straight up
    up: ArrayLike = field(default_factory=lambda: [0.0, 1.0, 0.0])
    # linear RGB radiance from straight up and straight down
    sky: ArrayLike = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ground: ArrayLike = field(default_factory=lambda: [0.0, 0.0, 0.0])
    softboxes: list = field(default_factory=list)

    def __post_init__(self):
        """
        Store plain floats and `Softbox` objects so settings compare
        equal after a round trip through JSON.
        """
        self.up = [float(value) for value in np.reshape(self.up, 3)]
        self.sky = [float(value) for value in np.reshape(self.sky, 3)]
        self.ground = [float(value) for value in np.reshape(self.ground, 3)]
        boxes = []
        for box in self.softboxes:
            if isinstance(box, dict):
                box = Softbox(**box)
            boxes.append(box)
        self.softboxes = boxes

    def light(self, direction: ArrayLike, blur: Number | NDArray) -> NDArray[float64]:
        """
        The environment's light around a direction averaged by a
        spherical Gaussian kernel which integrates to one, which is
        exactly what the OpenGL viewer's shader evaluates.

        Parameters
        ------------
        direction : (3,) or (n, 3) float
          Unit vectors to look at the environment in.
        blur : float or (n,) float
          Sharpness of the averaging kernel, `DIFFUSE_SHARPNESS` for
          what a diffuse surface facing `direction` receives.

        Returns
        ------------
        light : (3,) or (n, 3) float
          Linear RGB radiance in each direction.
        """
        direction = util.unitize(direction)
        # the gradient is linear in height so averaging it only scales
        # the height by the kernel's mean cosine
        spread = np.exp(-2.0 * blur)
        mean_cosine = (1.0 + spread) / (1.0 - spread) - 1.0 / blur
        height = np.dot(direction, util.unitize(self.up)) * mean_cosine
        sky, ground = np.array(self.sky), np.array(self.ground)
        light = ground + (sky - ground) * (0.5 + 0.5 * height)[..., None]
        kernel_peak = blur / (2.0 * np.pi * (1.0 - spread))
        for box in self.softboxes:
            overlap = gaussian_overlap(
                util.unitize(box.direction), box.sharpness, direction, blur
            )
            light = light + np.array(box.color) * (kernel_peak * overlap)[..., None]
        return light


@dataclass
class Studio:
    """
    Everything a renderer draws besides the scene's geometry with every
    default filled in for one camera, which is never stored on a scene.
    """

    environment: Environment
    # the scene's own lights, or the default key and rim lights
    lights: list
    # (n, 4, 4) world pose of each light right now
    transforms: NDArray
    # (4, 3) corners of a floor which only shows shadows, or None
    ground: NDArray | None
    # how bright a white floor facing up looks with nothing shadowing it
    ground_light: Number
    # how far ambient occlusion looks for nearby surfaces
    occlusion_radius: Number
    # the camera sees the environment behind the scene
    show_environment: bool = False
    # pixels across lines and points
    line_width: Number = 2.0
    point_size: Number = 6.0
    # what all light is multiplied by before it is tone mapped
    exposure: Number = 1.0
    # how much light too bright for the display spills onto its neighbors
    bloom: Number = 0.0


@dataclass
class RenderSettings:
    """
    How a scene is lit beyond its own lights. Store changed settings
    with `scene.metadata.update(settings.to_metadata())`.
    """

    # None is a photo studio arranged around the camera when rendering starts
    environment: Environment | dict | None = None
    # an invisible floor under the scene which only shows shadows
    ground_shadow: bool = True
    # the camera sees the environment behind the scene rather than a
    # plain background, like a sky or a starfield
    show_environment: bool = False
    # pixels across lines and points
    line_width: Number = 2.0
    point_size: Number = 6.0
    # what all light is multiplied by before it is tone mapped, where two
    # is twice as bright, which is one stop: lines, points and unlit
    # materials are colors rather than light, which it doesn't change
    exposure: Number = 1.0
    # how much of what the tone mapper had to compress towards white glows
    # onto the pixels around it, like glare in an eye or a lens, where zero
    # is none: only the OpenGL viewer draws it, Blender renders have none
    bloom: Number = 0.0

    def __post_init__(self):
        """
        Accept an environment written as a dict, as JSON stores it.
        """
        if isinstance(self.environment, dict):
            self.environment = Environment(**self.environment)
        self.ground_shadow = bool(self.ground_shadow)
        self.show_environment = bool(self.show_environment)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "RenderSettings":
        """
        Read settings stored by `to_metadata`, or the defaults.

        Parameters
        ------------
        metadata : dict
          Usually `scene.metadata`.

        Returns
        ------------
        settings : RenderSettings
          The stored settings or the defaults if there aren't any.
        """
        try:
            return cls(**metadata["render"])
        except (KeyError, TypeError):
            return cls()

    def to_metadata(self) -> dict:
        """
        The settings as JSON for `scene.metadata`, which is empty for
        the defaults so files don't grow a copy of them.

        Returns
        ------------
        metadata : dict
          `{"render": settings}` or `{}` for the defaults.
        """
        if self == RenderSettings():
            return {}
        return {"render": asdict(self)}

    def resolve(self, scene, camera_transform=None) -> Studio:
        """
        Fill in every default for rendering a scene from a camera: the
        default lights are a key and rim light fixed in the world
        relative to that camera, the default environment is a photo
        studio around it, and the floor is under the scene.

        Parameters
        ------------
        scene : trimesh.Scene
          Scene being rendered, which is not changed.
        camera_transform : None or (4, 4) float
          Where the camera is when rendering starts, defaulting to
          the scene's camera or the camera a render would frame it with.

        Returns
        ------------
        studio : Studio
          Concrete lights, environment and floor.
        """
        if camera_transform is None:
            from .cameras import current_view

            camera_transform = current_view(scene)[1]
        camera_transform = np.asarray(camera_transform, dtype=np.float64)
        if self.environment is None:
            # the world axis nearest to up on screen
            axes = np.vstack((np.eye(3), -np.eye(3)))
            up = axes[np.argmax(axes @ camera_transform[:3, 1])]
        else:
            up = util.unitize(self.environment.up)
        # level the camera's backwards direction, or for a camera looking
        # straight down the screen's down, then the right hand rule
        # completes the studio's frame
        levels = np.array([camera_transform[:3, 2], -camera_transform[:3, 1]])
        levels -= np.outer(levels @ up, up)
        back = util.unitize(levels[np.argmax(np.linalg.norm(levels, axis=1) > 1e-6)])
        right = np.cross(up, back)

        environment = self.environment
        if environment is None:
            environment = studio_environment(right=right, up=up, back=back)
        if scene.has_lights:
            lights = [light for light in scene.lights if light.name in scene.graph.nodes]
            transforms = [scene.graph.get(light.name)[0] for light in lights]
        else:
            lights, transforms = studio_lights(scene, right=right, up=up, back=back)
        transforms = np.reshape(transforms, (-1, 4, 4))

        bounds = scene.bounds
        ground = None
        if self.ground_shadow and bounds is not None:
            # a floor under the lowest corner of the scene
            center = bounds.mean(axis=0)
            height = (corners(bounds) @ up).min() - 1e-4 * scene.scale
            base = center + (height - np.dot(center, up)) * up
            # counterclockwise seen from above so it faces up
            square = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float64)
            ground = base + 4.0 * scene.scale * square @ np.vstack((right, back))

        # a white floor facing up reflects the directional lights and the
        # environment, which is what a shadow on it is darker than
        received = environment.light(up, DIFFUSE_SHARPNESS)
        for light, transform in zip(lights, transforms):
            if isinstance(light, DirectionalLight):
                facing = max(np.dot(up, util.unitize(transform[:3, 2])), 0.0)
                received = (
                    received + light.color[:3] / 255.0 * light.intensity * facing / np.pi
                )
        scale = 1.0
        if bounds is not None:
            scale = scene.scale
        return Studio(
            environment=environment,
            lights=lights,
            transforms=transforms,
            ground=ground,
            ground_light=float(np.dot(received, LUMINANCE)),
            occlusion_radius=OCCLUSION_RADIUS * scale,
            show_environment=self.show_environment,
            line_width=float(self.line_width),
            point_size=float(self.point_size),
            exposure=float(self.exposure),
            bloom=float(self.bloom),
        )


def specular_sharpness(roughness):
    """
    How blurry the environment looks reflected in a surface, as the
    sharpness of a spherical Gaussian as wide as the GGX lobe.

    Parameters
    ------------
    roughness : float or (n,) float
      glTF roughness from 0.04 to 1.

    Returns
    ------------
    sharpness : float or (n,) float
      Sharpness to pass to `Environment.light`.
    """
    return np.exp(np.polyval(SPECULAR_SHARPNESS, np.log(roughness)))


def gaussian_overlap(axis_a, sharpness_a, axis_b, sharpness_b):
    """
    The integral over the sphere of the product of two spherical
    Gaussians with peaks of one, written so it stays accurate when one
    of them is very sharp.

    Parameters
    ------------
    axis_a : (3,) float
      Unit vector of the first Gaussian.
    sharpness_a : float
      Sharpness of the first Gaussian.
    axis_b : (3,) or (n, 3) float
      Unit vectors of the second Gaussian.
    sharpness_b : float or (n,) float
      Sharpness of the second Gaussian.

    Returns
    ------------
    overlap : float or (n,) float
      The integral for each of `axis_b`.
    """
    combined = np.linalg.norm(
        sharpness_a * np.asarray(axis_a)
        + np.asarray(sharpness_b)[..., None] * np.asarray(axis_b),
        axis=-1,
    )
    total = combined + sharpness_a + sharpness_b
    exponent = 2.0 * sharpness_a * sharpness_b * (np.dot(axis_b, axis_a) - 1.0) / total
    limited = np.maximum(combined, 1e-6)
    return 2.0 * np.pi * np.exp(exponent) * (1.0 - np.exp(-2.0 * combined)) / limited


def with_luminance(color: ArrayLike, luminance: Number) -> NDArray[float64]:
    """
    Scale a color to a Rec.709 luminance, so tinting a light changes
    its color but not how bright it looks.

    Parameters
    ------------
    color : (3,) float
      Linear RGB in any units.
    luminance : float
      How bright the result looks.

    Returns
    ------------
    scaled : (3,) float
      Linear RGB with `luminance`.
    """
    color = np.asarray(color, dtype=np.float64)
    return color * (luminance / np.dot(color, LUMINANCE))


def studio_direction(right, up, back, azimuth: Number, elevation: Number):
    """
    A unit vector in a studio's frame.

    Parameters
    ------------
    right : (3,) float
      The studio's right.
    up : (3,) float
      The studio's up.
    back : (3,) float
      Towards the camera, level with the floor.
    azimuth : float
      Degrees around up from `back` towards `right`.
    elevation : float
      Degrees above the floor.

    Returns
    ------------
    direction : (3,) float
      Unit vector.
    """
    azimuth, elevation = np.radians(azimuth), np.radians(elevation)
    level = np.sin(azimuth) * np.asarray(right) + np.cos(azimuth) * np.asarray(back)
    return util.unitize(np.cos(elevation) * level + np.sin(elevation) * np.asarray(up))


def studio_environment(right, up, back) -> Environment:
    """
    The default environment: a dim gradient from a dark floor to a
    pale sky, a broad light overhead and two softboxes either side of
    the camera which put window-like highlights on shiny surfaces.

    Parameters
    ------------
    right : (3,) float
      The studio's right.
    up : (3,) float
      The studio's up.
    back : (3,) float
      Towards the camera, level with the floor.

    Returns
    ------------
    environment : Environment
      In world space.
    """
    return Environment(
        up=up,
        sky=with_luminance(SKY_COLOR, 0.31),
        ground=with_luminance(SKY_COLOR, 0.096),
        softboxes=[
            Softbox(direction=up, sharpness=4.0, color=[0.35] * 3),
            # warm like the key light on its side and cool opposite
            Softbox(
                direction=studio_direction(right, up, back, 60.0, 20.0),
                sharpness=30.0,
                color=with_luminance(KEY_COLOR, 2.0),
            ),
            Softbox(
                direction=studio_direction(right, up, back, -70.0, 12.0),
                sharpness=30.0,
                color=with_luminance(RIM_COLOR, 1.0),
            ),
        ],
    )


def studio_lights(scene, right, up, back) -> tuple[list, NDArray[float64]]:
    """
    The default lights: a warm key light from in front, to the right
    and above, and a cool rim light from high behind which outlines
    things, high so its shadow stays short and under the scene.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene the lights are for, only used to name them uniquely.
    right : (3,) float
      The studio's right.
    up : (3,) float
      The studio's up.
    back : (3,) float
      Towards the camera, level with the floor.

    Returns
    ------------
    lights : (2,) DirectionalLight
      The key and rim lights.
    transforms : (2, 4, 4) float
      World pose of each light, which shines down its -Z.
    """
    names = set(scene.graph.nodes)
    # in glTF units a white surface facing a directional light of intensity
    # `I` lux and linear color `C` reflects `C * I / pi`, which for these is
    # a luminance of 0.58 and 0.28
    lights = [
        DirectionalLight(
            name=util.unique_name("key_light", names),
            color=[*KEY_COLOR, 255],
            intensity=np.pi * 0.58 / np.dot(KEY_COLOR / 255.0, LUMINANCE),
        ),
        DirectionalLight(
            name=util.unique_name("rim_light", names),
            color=[*RIM_COLOR, 255],
            intensity=np.pi * 0.28 / np.dot(RIM_COLOR / 255.0, LUMINANCE),
        ),
    ]
    towards = [
        studio_direction(right, up, back, 45.0, 35.0),
        studio_direction(right, up, back, 210.0, 65.0),
    ]
    # a light's +Z points back towards where its light comes from
    transforms = np.array([align_vectors([0.0, 0.0, 1.0], vector) for vector in towards])
    return lights, transforms
