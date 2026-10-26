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

from .. import util
from ..base import Trimesh
from ..bounds import corners as bounds_corners
from ..path.path import Path
from ..points import PointCloud
from ..scene.cameras import projection_matrices
from ..scene.lighting import (
    HARMONICS,
    DirectionalLight,
    EnvironmentLight,
    SpotLight,
    autolight,
    prefilter,
    spherical_harmonics,
)
from ..typed import ArrayLike, Integer, NDArray, float64
from ..visual.color import to_rgba
from ..visual.material import PBRMaterial
from ..voxel.base import VoxelGrid
from .opengl.shaders import (
    ALPHA_MODES,
    FRAME,
    LUMINANCE,
    MATERIAL,
    MATERIAL_TEXTURES,
    MAX_LIGHTS,
    MAX_SHADOWS,
    SEGMENT,
    SKIN,
    VERTEX,
)

# How every renderer draws a scene unless the keywords of `SceneViewer`,
# `Scene.show`, `Scene.render` or `Scene.save_image` say otherwise, which
# are never stored in the scene so its files only hold the scene itself
RENDER_DEFAULTS = {
    # an invisible floor under the scene which only shows shadows
    "ground_shadow": True,
    # the camera sees the environment behind the scene, like a sky,
    # rather than a plain background
    "show_environment": False,
    # pixels across lines and points
    "line_width": 2.0,
    "point_size": 6.0,
    # what light is multiplied by before tone mapping, where two is one stop
    # brighter: lines, points and unlit materials are colors, not light
    "exposure": 1.0,
    # how much of what tone mapping compressed towards white glows onto the
    # pixels around it like glare, which only the OpenGL viewer draws
    "bloom": 0.0,
}

# the radius ambient occlusion looks for nearby surfaces within, as a
# fraction of `scene.scale` which is the length of the bounds diagonal
OCCLUSION_RADIUS = 0.1


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
    floor is, how far ambient occlusion looks, and the render settings.

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
    # a missing range is NaN which `nan_to_num` makes zero: reaches forever
    radius = np.array([item.radius for item in placed], dtype=np.float64)
    inner = np.array([getattr(item, "innerConeAngle", 0.0) for item in placed])
    outer = np.array([getattr(item, "outerConeAngle", 0.0) for item in placed])
    colors = np.reshape([item.color[:3] for item in placed], (-1, 3)) / 255.0
    cone_scale = np.where(spot, 1.0 / np.maximum(np.cos(inner) - np.cos(outer), 1e-3), 0)
    casts = directional | spot
    rows = frame["lights"][0, : len(placed)]
    rows["position"][:, 3] = ~directional
    rows["color"][:, :3] = colors * intensity[:, None]
    rows["falloff"][:, 0] = np.where(directional, 0.0, np.nan_to_num(1.0 / radius))
    rows["falloff"][:, 1] = cone_scale
    rows["falloff"][:, 2] = np.where(spot, -np.cos(outer) * cone_scale, 1.0)
    rows["falloff"][:, 3] = np.where(spot, outer, 0.0)
    order = np.cumsum(casts) - 1
    rows["shadow"][:, 0] = np.where(casts & (order < MAX_SHADOWS), order, -1)
    frame["light_count"][0, 0] = len(placed)

    # world to the environment's frame, which a draw makes view to it, and
    # the irradiance of a white diffuse surface over pi so the shader only
    # multiplies the polynomials of `HARMONICS`
    transforms = np.asarray(transforms)
    light, rotation = lights[-1], util.unitize(transforms[-1, :3, :3].T)
    tint, irradiance = light.intensity, spherical_harmonics(light.cubemap)
    frame["environment_rotation"][0, :3, :3] = rotation
    frame["environment_color"][0] = (tint, tint, tint, len(prefilter(light.cubemap)) - 1)
    frame["irradiance"][0, :, :3] = irradiance * HARMONICS[:, None] * tint / np.pi
    # how bright a white floor facing up looks unshadowed: the environment's
    # `HARMONICS` at +Y, and `C * I / pi` from each directional light
    facing = util.unitize(transforms[: len(placed), :3, 2]) @ rotation[1]
    received = (HARMONICS * [1, 1, 0, 0, 0, 0, -1, 0, -1]) @ irradiance * tint
    received += (np.clip(facing, 0.0, None) * directional * intensity) @ colors
    frame["ground_light"][0, 0] = received @ LUMINANCE / np.pi
    # a fraction of the scene's size, or of one if it's empty
    frame["occlusion_radius"][0, 0] = OCCLUSION_RADIUS
    if scene.bounds is not None:
        frame["occlusion_radius"][0, 0] *= scene.scale
    frame["exposure"][0, 0] = settings["exposure"]
    frame["line_widths"][0, :2] = (settings["line_width"], settings["point_size"])
    return frame


def geometry_to_arrays(geometry) -> GeometryArrays | None:
    """
    Convert any supported trimesh geometry into `GeometryArrays`. Points
    are zero length segments which the line shader draws as dots, and
    paths are one segment per discretized piece of every entity, colored
    like its entity. A mesh is split along sharp edges by
    `mesh.smooth_shaded`, which keeps faces in order so face colors still
    line up with `gl_PrimitiveID`, with its material as the record the
    mesh shader reads with glTF defaults for anything not set, cached on
    the mesh until it or its visual changes so it is shared and never to
    be changed.

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
    if not isinstance(geometry, Trimesh):
        if isinstance(geometry, PointCloud):
            kind, colors = "points", geometry.colors
            start = end = util.stack_3D(geometry.vertices)
        elif isinstance(geometry, Path):
            kind, colors = "lines", geometry.colors
            if colors is not None:
                colors = colors[geometry.segments_entity]
            # 2D paths are padded to Z=0
            segments = geometry.segments
            vertices = util.stack_3D(segments.reshape((-1, segments.shape[2])))
            start, end = vertices[0::2], vertices[1::2]
        else:
            util.log.warning(f"unable to draw `{type(geometry).__name__}`")
            return None
        rows = np.zeros(len(start), dtype=SEGMENT)
        rows["segment_start"], rows["segment_end"] = start, end
        # lines and points without colors are drawn in black
        rows["color"] = (0, 0, 0, 255)
        if colors is not None and len(colors) > 0:
            rows["color"] = to_rgba(colors)
        return GeometryArrays(kind=kind, rows=rows, material=np.zeros((), MATERIAL))

    # keyed like `smooth_shaded` by the visual, and by the material which
    # isn't part of its hash, in the mesh cache which is cleared whenever
    # the vertices or faces change
    material = getattr(geometry.visual, "material", None)
    key = f"viewer_arrays_{hash(geometry.visual)}_{hash(material)}"
    if geometry._cache[key] is not None:
        return geometry._cache[key]
    smooth = geometry.smooth_shaded
    visual = smooth.visual
    # every mesh gets every attribute so the shader never branches:
    # the defaults are white colors and zero uv
    rows = np.zeros(len(smooth.vertices), dtype=VERTEX)
    rows["position"], rows["normal"] = smooth.vertices, smooth.vertex_normals
    rows["color"] = 255
    # without a material a mesh is matte grey plastic, not glTF's metal
    face_colors, pbr = None, PBRMaterial(metallicFactor=0.0, roughnessFactor=0.5)
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

    # every factor the shader reads, or its glTF default if it isn't set
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
    for name in value:
        if getattr(pbr, name) is not None:
            value[name] = getattr(pbr, name)
    record = np.zeros((), dtype=MATERIAL)
    record["base_color"] = np.asarray(value["baseColorFactor"]) / 255.0
    record["emissive"][:3] = value["emissiveFactor"]
    record["factors"][:2] = value["metallicFactor"], value["roughnessFactor"]
    record["factors"][2:] = value["alphaCutoff"], value["diffuseTransmissionFactor"]
    record["transmission_color"][:3] = value["diffuseTransmissionColorFactor"]
    record["transmission_color"][3] = value["transmissionFactor"]
    alpha_mode = ALPHA_MODES[value["alphaMode"]]
    record["flags"] = (alpha_mode, bool(pbr.unlit), 0, bool(pbr.doubleSided))
    # the textures by their name in the shader, which is glTF's in snake case
    textures = {}
    for name in MATERIAL_TEXTURES:
        first, *rest = name.split("_")
        image = getattr(pbr, first + "".join(word.title() for word in rest))
        if image is not None:
            textures[name] = image
    # glass, which is BLEND as `ALPHA_MODES` describes, and translucent
    # colors which blend even if the material doesn't
    alpha = rows["color"][:, 3]
    if face_colors is not None:
        alpha = face_colors[:, 3]
    if pbr.transmissionFactor is not None or (alpha < 255).any():
        record["flags"][0] = max(record["flags"][0], ALPHA_MODES["BLEND"])
    record["has_textures"][:2] = ("normal_texture" in textures, face_colors is not None)

    # the joints and weights of a skinned mesh, which `smooth_shaded` keeps
    skin, attributes = None, smooth.vertex_attributes
    if attributes.get("joints") is not None and attributes.get("weights") is not None:
        skin = np.zeros(len(rows), dtype=SKIN)
        skin["joints"], skin["weights"] = attributes["joints"], attributes["weights"]

    arrays = GeometryArrays(
        kind="mesh",
        rows=rows,
        material=record,
        faces=np.asarray(smooth.faces, dtype=np.uint32),
        skin=skin,
        face_colors=face_colors,
        textures=textures,
    )
    geometry._cache[key] = arrays
    return arrays


def shadow_projections(
    world: ArrayLike,
    directional: ArrayLike,
    cone_angles: ArrayLike,
    bounds: ArrayLike,
    resolution: Integer,
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
    directional = np.asarray(directional, dtype=bool)
    # a light's node may have a scale, which isn't part of the light, and
    # without it the light's space and its texels are in world units
    world[:, :3, :3] /= np.linalg.norm(world[:, :3, :3], axis=1, keepdims=True)
    to_light = np.linalg.inv(world)
    corners = bounds_corners(bounds)
    local = np.matmul(corners, np.swapaxes(to_light[:, :3, :3], 1, 2))
    local += to_light[:, None, :3, 3]
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
    nearest = depth.min(axis=1)
    near = np.where(directional, nearest - margin, np.maximum(nearest, far * 1e-3))
    # a cone between straight ahead and a hemisphere has a finite focal length
    focal = 1.0 / np.tan(np.clip(cone_angles, 1e-3, np.pi / 2.0 - 1e-3))
    scale = np.where(directional[:, None], 2.0 / size, focal[:, None])
    projection = projection_matrices(scale, near, far, directional)
    widths = np.where(directional, size.max(axis=1), 2.0 / focal)
    texels = np.column_stack((widths * directional, widths * ~directional))
    return np.matmul(projection, to_light), texels / float(resolution)


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
    studio = list(zip(*autolight(scene, camera_transform)))
    pairs = studio
    if scene.has_lights:
        pairs = [
            (light, scene.graph.get(light.name)[0])
            for light in scene.lights
            if light.name in scene.graph.nodes
        ]
    environments = [pair for pair in pairs if isinstance(pair[0], EnvironmentLight)]
    if len(environments) > 1:
        util.log.warning("rendering the first of several environment lights")
    # the punctual lights then one environment, the studio's if there's none
    pairs = [pair for pair in pairs if not isinstance(pair[0], EnvironmentLight)]
    lights, transforms = zip(*pairs, [*environments, studio[-1]][0])
    return list(lights), np.reshape(transforms, (-1, 4, 4))
