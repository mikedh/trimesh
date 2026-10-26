"""
conversion.py
---------------

Convert trimesh geometry and lights into the numpy structured arrays
declared in `opengl/shaders` which are uploaded to the GPU as they are.
Nothing here imports pyglet or calls OpenGL, so all of it can be tested
anywhere.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .. import caching, util
from ..base import Trimesh
from ..bounds import corners as bounds_corners
from ..path.path import Path
from ..points import PointCloud
from ..scene.cameras import projection_matrices
from ..scene.lighting import (
    HARMONICS,
    LUMINANCE,
    STUDIO_SHARPNESS,
    DirectionalLight,
    EnvironmentLight,
    SpotLight,
    autolight,
    cubemap_directions,
    cubemap_sample,
    cubemap_size,
    spherical_harmonics,
    studio_environment,
)
from ..typed import ArrayLike, NDArray, float64
from ..visual.color import to_rgba
from ..visual.material import PBRMaterial
from ..voxel.base import VoxelGrid
from .opengl.shaders import (
    ALPHA_MODES,
    FRAME,
    MATERIAL,
    MATERIAL_TEXTURES,
    MAX_LIGHTS,
    MAX_SHADOWS,
    SEGMENT,
    SKIN,
    SPECULAR_SHARPNESS,
    VERTEX,
)

# lines and points without colors are drawn in black
DEFAULT_LINE_COLOR = np.array([0, 0, 0, 255], dtype=np.uint8)

# the glTF material of meshes without one, which is a matte grey plastic
# rather than the glTF default of a metal
PLASTIC = {"metallicFactor": 0.0, "roughnessFactor": 0.5}

# How every renderer draws a scene unless the keywords of `SceneViewer`,
# `Scene.show`, `Scene.render` or `Scene.save_image` say otherwise, which
# are never stored in the scene so its files only hold the scene itself
RENDER_DEFAULTS = {
    # an invisible floor under the scene which only shows shadows
    "ground_shadow": True,
    # the camera sees the environment behind the scene rather than a
    # plain background, like a sky or a starfield
    "show_environment": False,
    # pixels across lines and points
    "line_width": 2.0,
    "point_size": 6.0,
    # what all light is multiplied by before it is tone mapped, where two
    # is twice as bright, which is one stop: lines, points and unlit
    # materials are colors rather than light, which it doesn't change
    "exposure": 1.0,
    # how much of what the tone mapper had to compress towards white glows
    # onto the pixels around it, like glare in an eye or a lens, where zero
    # is none: only the OpenGL viewer draws it, Blender renders have none
    "bloom": 0.0,
}

# the radius ambient occlusion looks for nearby surfaces within, as a
# fraction of `scene.scale` which is the length of the bounds diagonal
OCCLUSION_RADIUS = 0.1

# texels across each face of the cube map every environment is drawn
# from, as the default studio's: a mip level down still holds its
# sharpest softbox, which reflections blend in as roughness is never zero
ENVIRONMENT_SIZE = 2 * cubemap_size(STUDIO_SHARPNESS.max())
# its mip levels, down to the one which holds the lobe of roughness one,
# whose sharpness is the constant of the fit as `log(1)` is zero
ENVIRONMENT_LEVELS = 1 + int(
    np.log2(ENVIRONMENT_SIZE // cubemap_size(np.exp(SPECULAR_SHARPNESS[-1])))
)

# `environment_levels` by the hash of the cube map they were made from
_ENVIRONMENT_LEVELS = {}


@dataclass
class GeometryArrays:
    """
    Everything needed to draw one piece of geometry as the structured
    arrays the GPU reads: indexed triangles, or line segments where a
    point is a segment with two identical endpoints.
    """

    kind: Literal["mesh", "lines", "points"]
    # `shaders.VERTEX` rows for a mesh, `shaders.SEGMENT` rows otherwise
    rows: NDArray
    # one `shaders.MATERIAL` record
    material: NDArray
    # mesh only: (m, 3) uint32
    faces: NDArray | None = None
    # mesh only: `shaders.SKIN` rows, or None for a mesh without a skin
    skin: NDArray | None = None
    # mesh only: (m, 4) uint8 or None if colors are per-vertex
    face_colors: NDArray | None = None
    # `shaders.MATERIAL_TEXTURES` name to `PIL.Image.Image`, for the ones set
    textures: dict = field(default_factory=dict)


def frame_constants(
    scene, lights: list, transforms: ArrayLike, settings: dict
) -> NDArray:
    """
    What every frame of a render starts from: the `KHR_lights_punctual`
    lights with everything which only changes with the scene, where the
    spot cone is precomputed as the spec suggests so every kind of light
    is the same math, the environment in world space, how bright the
    floor is, how far ambient occlusion looks, and the exposure.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene being drawn.
    lights : list of trimesh.scene.lighting.Light
      From `render_lights`, which ends with the environment.
    transforms : (n, 4, 4) float
      World pose of each light.
    settings : dict
      `RENDER_DEFAULTS` with any changed by keywords.

    Returns
    ------------
    frame : (1,) shaders.FRAME
      With only those filled in, where the first `MAX_SHADOWS`
      directional and spot lights get a shadow map layer.
    """
    frame = np.zeros(1, dtype=FRAME)
    placed = lights[:-1]
    if len(placed) > MAX_LIGHTS:
        util.log.warning(f"drawing {MAX_LIGHTS} of {len(placed)} lights")
        placed = placed[:MAX_LIGHTS]
    directional = np.array([isinstance(item, DirectionalLight) for item in placed], bool)
    spot = np.array([isinstance(item, SpotLight) for item in placed], dtype=bool)
    intensity = np.array([item.intensity for item in placed], dtype=np.float64)
    colors = np.reshape([item.color[:3] for item in placed], (-1, 3)) / 255.0
    # a missing range is NaN which `nan_to_num` makes zero: reaches forever
    radius = np.array([item.radius for item in placed], dtype=np.float64)
    inner = np.array([getattr(item, "innerConeAngle", 0.0) for item in placed])
    outer = np.array([getattr(item, "outerConeAngle", 0.0) for item in placed])
    cone_scale = np.where(spot, 1.0 / np.maximum(np.cos(inner) - np.cos(outer), 1e-3), 0)
    casts = directional | spot
    order = np.cumsum(casts) - 1
    rows = frame["lights"][0, : len(placed)]
    rows["position"][:, 3] = ~directional
    rows["color"][:, :3] = colors * intensity.reshape((-1, 1))
    rows["falloff"][:, 0] = np.where(directional, 0.0, np.nan_to_num(1.0 / radius))
    rows["falloff"][:, 1] = cone_scale
    rows["falloff"][:, 2] = np.where(spot, -np.cos(outer) * cone_scale, 1.0)
    rows["falloff"][:, 3] = np.where(spot, outer, 0.0)
    rows["shadow"][:, 0] = np.where(casts & (order < MAX_SHADOWS), order, -1)
    frame["light_count"][0, 0] = len(placed)

    transforms = np.asarray(transforms)
    light, rotation = lights[-1], util.unitize(transforms[-1, :3, :3].T)
    tint = light.color[:3] / 255.0 * light.intensity
    # world to the environment's frame, which a draw makes view to it,
    # and the irradiance of a white diffuse surface over pi so the shader
    # only multiplies the polynomials of `spherical_harmonics`
    frame["environment_rotation"][0, :3, :3] = rotation
    frame["environment_color"][0] = (*tint, ENVIRONMENT_LEVELS - 1)
    frame["irradiance"][0, :, :3] = light.irradiance * HARMONICS[:, None] * tint / np.pi
    up = rotation[1]
    # how bright a white floor facing up looks with nothing shadowing it,
    # which a shadow on the floor is darker than: what it gets from the
    # environment and from directional lights, which a white surface
    # facing one reflects `C * I / pi` of
    facing = np.clip(util.unitize(transforms[: len(placed), :3, 2]) @ up, 0.0, None)
    received = spherical_harmonics(np.eye(3)[1]) @ light.irradiance * tint / np.pi
    received = received + (facing * directional * intensity) @ colors / np.pi
    frame["ground_light"][0, 0] = received @ LUMINANCE
    # a fraction of the scene's size, or of one if it's empty
    frame["occlusion_radius"][0, 0] = OCCLUSION_RADIUS
    if scene.bounds is not None:
        frame["occlusion_radius"][0, 0] *= scene.scale
    frame["exposure"][0, 0] = settings["exposure"]
    return frame


def geometry_to_arrays(geometry) -> GeometryArrays | None:
    """
    Convert any supported trimesh geometry into `GeometryArrays`, where
    points are zero length segments which the line shader draws as dots
    and paths are one segment per discretized piece of every entity,
    colored like its entity.

    Parameters
    ------------
    geometry : Trimesh, PointCloud, Path2D, Path3D or VoxelGrid
      Geometry to convert.

    Returns
    ------------
    arrays : GeometryArrays or None
      Arrays for the GPU, or None if the geometry is empty or unsupported.
    """
    if isinstance(geometry, VoxelGrid):
        geometry = geometry.as_boxes()
    if geometry.is_empty:
        return None
    if isinstance(geometry, Trimesh):
        return mesh_to_arrays(geometry)
    if isinstance(geometry, PointCloud):
        kind, colors = "points", geometry.colors
        start = end = util.stack_3D(geometry.vertices)
    elif isinstance(geometry, Path):
        kind, colors = "lines", geometry.colors
        if colors is not None:
            colors = colors[geometry.segments_entity]
        segments = geometry.segments
        # 2D paths are padded to Z=0
        vertices = util.stack_3D(segments.reshape((-1, segments.shape[2])))
        start, end = vertices[0::2], vertices[1::2]
    else:
        util.log.warning(f"unable to draw `{type(geometry).__name__}`")
        return None
    rows = np.zeros(len(start), dtype=SEGMENT)
    rows["segment_start"], rows["segment_end"] = start, end
    rows["color"] = DEFAULT_LINE_COLOR
    if colors is not None and len(colors) > 0:
        rows["color"] = to_rgba(colors)
    return GeometryArrays(kind=kind, rows=rows, material=np.zeros((), MATERIAL))


def mesh_to_arrays(mesh: Trimesh) -> GeometryArrays:
    """
    Convert a mesh to indexed triangle rows and its material to the
    record the mesh shader reads, with glTF defaults for anything not
    set. The mesh is split along sharp edges with `mesh.smooth_shaded`,
    which keeps faces in order so face colors still line up with
    `gl_PrimitiveID`. The result is cached on the mesh until it or its
    visual changes, so it is shared and never to be changed.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Mesh to convert.

    Returns
    ------------
    arrays : GeometryArrays
      Indexed triangles with per-vertex normals, uv and colors.
    """
    # keyed like `smooth_shaded` by the visual, and by the material which
    # isn't part of its hash, in the mesh cache which is cleared whenever
    # the vertices or faces change
    material = getattr(mesh.visual, "material", None)
    key = f"viewer_arrays_{hash(mesh.visual)}_{hash(material)}"
    cached = mesh._cache[key]
    if cached is not None:
        return cached

    smooth = mesh.smooth_shaded
    visual = smooth.visual
    # every mesh gets every attribute so the shader never branches:
    # the defaults are white colors and zero uv
    rows = np.zeros(len(smooth.vertices), dtype=VERTEX)
    rows["position"] = smooth.vertices
    rows["normal"] = smooth.vertex_normals
    rows["color"] = 255
    face_colors = None
    pbr = PBRMaterial(**PLASTIC)
    if visual.kind == "texture":
        pbr = visual.material.to_pbr()
        if visual.uv is not None and len(visual.uv) == len(rows):
            rows["uv"] = visual.uv[:, :2]
        # per-vertex colors multiply the material like glTF `COLOR_0`
        colors = visual.vertex_attributes.get("color")
        if colors is not None and len(colors) == len(rows):
            rows["color"] = to_rgba(colors)
    elif visual.kind == "face":
        face_colors = np.asarray(visual.face_colors, dtype=np.uint8)
    else:
        # vertex colors, or the default color if nothing is set
        rows["color"] = visual.vertex_colors

    # the glTF default for every factor which isn't set
    value = {
        "baseColorFactor": [255, 255, 255, 255],
        "emissiveFactor": [0.0, 0.0, 0.0],
        "metallicFactor": 1.0,
        "roughnessFactor": 1.0,
        "alphaCutoff": 0.5,
        "diffuseTransmissionFactor": 0.0,
        "diffuseTransmissionColorFactor": [1.0, 1.0, 1.0],
        "transmissionFactor": 0.0,
        "alphaMode": "OPAQUE",
    }
    for name in list(value):
        if getattr(pbr, name) is not None:
            value[name] = getattr(pbr, name)
    record = np.zeros((), dtype=MATERIAL)
    record["base_color"] = np.asarray(value["baseColorFactor"]) / 255.0
    record["emissive"][:3] = value["emissiveFactor"]
    record["factors"] = [
        value[name]
        for name in (
            "metallicFactor",
            "roughnessFactor",
            "alphaCutoff",
            "diffuseTransmissionFactor",
        )
    ]
    record["transmission_color"][:3] = value["diffuseTransmissionColorFactor"]
    alpha_mode = ALPHA_MODES[value["alphaMode"]]
    record["flags"] = (alpha_mode, bool(pbr.unlit), 0, bool(pbr.doubleSided))
    if pbr.transmissionFactor is not None:
        # glass, which is BLEND as `ALPHA_MODES` describes
        record["base_color"][3] *= 1.0 - value["transmissionFactor"]
        record["flags"][0] = ALPHA_MODES["BLEND"]
    # the textures by their name in the shader from their glTF name
    images = {
        "base_color_texture": pbr.baseColorTexture,
        "metallic_roughness_texture": pbr.metallicRoughnessTexture,
        "emissive_texture": pbr.emissiveTexture,
        "normal_texture": pbr.normalTexture,
        "occlusion_texture": pbr.occlusionTexture,
        "diffuse_transmission_texture": pbr.diffuseTransmissionTexture,
        "diffuse_transmission_color_texture": pbr.diffuseTransmissionColorTexture,
    }
    textures = {
        name: images[name] for name in MATERIAL_TEXTURES if images[name] is not None
    }
    record["has_textures"][0] = "normal_texture" in textures
    # translucent colors blend even if the material doesn't
    colors = rows["color"]
    if face_colors is not None:
        colors = face_colors
        record["has_textures"][1] = 1
    if (colors[:, 3] < 255).any():
        record["flags"][0] = max(record["flags"][0], ALPHA_MODES["BLEND"])

    # the joints and weights of a skinned mesh, which `smooth_shaded` keeps
    skin = None
    joints = smooth.vertex_attributes.get("joints")
    weights = smooth.vertex_attributes.get("weights")
    if joints is not None and weights is not None:
        skin = np.zeros(len(rows), dtype=SKIN)
        skin["joints"] = joints
        skin["weights"] = weights

    arrays = GeometryArrays(
        kind="mesh",
        rows=rows,
        material=record,
        faces=np.asarray(smooth.faces, dtype=np.uint32),
        skin=skin,
        face_colors=face_colors,
        textures=textures,
    )
    mesh._cache[key] = arrays
    return arrays


def shadow_projections(
    world, directional, cone_angles, bounds, resolution: int
) -> tuple[NDArray, NDArray]:
    """
    Where each light sees the scene from, as one projection per light
    fit tightly around the scene's bounds so no shadow map resolution
    is wasted: an orthographic box for directional lights and the cone
    of spot lights.

    Parameters
    ------------
    world : (k, 4, 4) float
      Pose of each light, which shines down its -Z.
    directional : (k,) bool
      Lights which are infinitely far away.
    cone_angles : (k,) float
      Outer cone angle of each spot light.
    bounds : (2, 3) float
      Axis aligned bounds of everything casting shadows.
    resolution : int
      Width and height of the shadow maps in pixels.

    Returns
    ------------
    clip : (k, 4, 4) float
      World to clip space of each light.
    texels : (k, 2) float
      Width of a shadow map pixel in world units as a constant for
      directional lights plus a multiple of the distance from spot lights.
    """
    world = np.array(world, dtype=np.float64).reshape((-1, 4, 4))
    # a light's node may have a scale, which isn't part of the light, and
    # without it the light's space and its texels are in world units
    world[:, :3, :3] /= np.linalg.norm(world[:, :3, :3], axis=1, keepdims=True)
    to_light = np.linalg.inv(world)
    corners = bounds_corners(bounds)
    local = (
        np.matmul(corners, np.swapaxes(to_light[:, :3, :3], 1, 2))
        + to_light[:, None, :3, 3]
    )
    # distance in front of each light, which looks down its -Z
    depth = -local[..., 2]
    # the planes sit a thousandth of the scene outside of it, which is far
    # more than the float32 rounding of a corner's depth, so nothing is
    # clipped, plus a length so a scene of a single point has some depth
    margin = 1e-3 * np.ptp(corners, axis=0).max() + 1e-12
    far = depth.max(axis=1) + margin
    low, high = local[..., :2].min(axis=1), local[..., :2].max(axis=1)
    # as wide as the scene, and something for a scene seen edge on
    size = np.maximum(high - low, 1e-12)
    # a directional light looks from the middle of the scene's box
    to_light[directional, :2, 3] -= (high + low)[directional] / 2.0

    # a spot light can't see behind itself so its near plane is in front,
    # where a 24 bit depth buffer tells apart `far^2 / (near * 2^24)` at
    # the far plane, so the near plane is kept where that is under 0.01%
    near = np.where(
        directional,
        depth.min(axis=1) - margin,
        np.maximum(depth.min(axis=1), far * 1e-3),
    )
    # a cone between straight ahead and a hemisphere has a finite focal length
    focal = 1.0 / np.tan(np.clip(cone_angles, 1e-3, np.pi / 2.0 - 1e-3))
    scale = np.where(directional[:, None], 2.0 / size, focal[:, None])
    projection = projection_matrices(scale, near, far, directional)
    texels = np.column_stack(
        (
            np.where(directional, size.max(axis=1), 0.0),
            np.where(directional, 0.0, 2.0 / focal),
        )
    ) / float(resolution)
    return np.matmul(projection, to_light), texels


def render_lights(scene, camera_transform: ArrayLike) -> tuple[list, NDArray[float64]]:
    """
    The lights a render of a scene uses without changing the scene: its
    own lights, or `autolight` around where the camera starts if it has
    none, with the default environment if its own have no `EnvironmentLight`.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene being rendered, which is not changed.
    camera_transform : (4, 4) float
      Where the camera is when rendering starts.

    Returns
    ------------
    lights : list of trimesh.scene.lighting.Light
      Directional, point and spot lights, then one `EnvironmentLight`.
    transforms : (n, 4, 4) float
      World pose of each light right now.
    """
    environment, frame = studio_environment(scene, camera_transform)
    defaults, poses = autolight(scene, camera_transform)
    defaults, poses = [*defaults, environment], np.vstack((poses, frame[None]))
    if not scene.has_lights:
        return defaults, poses
    lights = [light for light in scene.lights if light.name in scene.graph.nodes]
    transforms = [scene.graph.get(light.name)[0] for light in lights]
    environments = [isinstance(light, EnvironmentLight) for light in lights]
    order = np.argsort(environments, kind="stable")
    lights, transforms = [lights[i] for i in order], [transforms[i] for i in order]
    if not any(environments):
        lights.append(defaults[-1])
        transforms.append(poses[-1])
    elif sum(environments) > 1:
        util.log.warning("rendering the first of several environment lights")
        first = len(lights) - sum(environments) + 1
        lights, transforms = lights[:first], transforms[:first]
    return lights, np.reshape(transforms, (-1, 4, 4))


def environment_levels(cubemap: NDArray) -> list[NDArray]:
    """
    The mip levels of a cube map the viewer reflects, laid out as
    `KHR_lights_environment` describes: level `k` of `n` is the radiance
    blurred by the spherical Gaussian as wide as a GGX reflection of
    roughness `k / (n - 1)`, from a mirror at level zero, for
    `ENVIRONMENT_LEVELS` from `ENVIRONMENT_SIZE`. Each level is the
    integral over every texel of the cube map averaged down to the
    level's size, cached as it only depends on the cube map.

    Parameters
    ------------
    cubemap : (6, n, n, 3) float
      Radiance of the faces of `lighting.CUBE_FACES`.

    Returns
    ------------
    levels : list of (6, m, m, 3) float32
      Every mip level, each half as wide as the one before.
    """
    cubemap = np.asarray(cubemap, dtype=np.float64)
    key = caching.hash_fast(cubemap.tobytes()) + hash(cubemap.shape)
    try:
        return _ENVIRONMENT_LEVELS[key]
    except KeyError:
        pass
    if cubemap.shape[1] % ENVIRONMENT_SIZE != 0:
        directions = cubemap_directions(ENVIRONMENT_SIZE)[0]
        cubemap = cubemap_sample(cubemap, directions)
    # float32 is twice as fast and rounds `d @ d.T - 1` under 1e-7
    cubemap = cubemap.astype(np.float32)
    levels = []
    for level in range(ENVIRONMENT_LEVELS):
        width = ENVIRONMENT_SIZE >> level
        step = cubemap.shape[1] // width
        blurred = cubemap.reshape((6, width, step, width, step, 3)).mean(axis=(2, 4))
        if level > 0:
            roughness = level / (ENVIRONMENT_LEVELS - 1)
            sharpness = np.float32(
                np.exp(np.polyval(SPECULAR_SHARPNESS, np.log(roughness)))
            )
            directions, solid_angle = cubemap_directions(width)
            directions = directions.reshape((-1, 3)).astype(np.float32)
            # BLAS multiplies by a contiguous transpose five times faster
            cosine = directions @ np.ascontiguousarray(directions.T)
            weights = np.exp(sharpness * (cosine - 1.0))
            weights *= solid_angle.reshape(-1)
            blurred = (weights @ blurred.reshape((-1, 3))) / weights.sum(axis=1)[:, None]
        levels.append(blurred.reshape((6, width, width, 3)))
    _ENVIRONMENT_LEVELS[key] = levels
    return levels
