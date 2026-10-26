"""
conversion.py
---------------

Convert trimesh geometry and lights into the numpy structured arrays
declared in `opengl/shaders` which are uploaded to the GPU as they are.
Nothing here imports pyglet or calls OpenGL, so all of it can be tested
anywhere and Blender renders can use it.
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
    LUMINANCE,
    DirectionalLight,
    EnvironmentLight,
    SpotLight,
    autolight,
)
from ..typed import ArrayLike, NDArray, Number, float64
from ..visual.color import to_rgba
from ..visual.material import PBRMaterial
from ..voxel.base import VoxelGrid
from .opengl.shaders import (
    ALPHA_MODES,
    DIFFUSE_SHARPNESS,
    FRAME,
    LIGHT,
    MATERIAL,
    MATERIAL_TEXTURES,
    MAX_SOFTBOXES,
    SEGMENT,
    SKIN,
    VERTEX,
)

# lines and points without colors are drawn in black
DEFAULT_LINE_COLOR = np.array([0, 0, 0, 255], dtype=np.uint8)

# the glTF material of meshes without one, which is a matte grey plastic
# rather than the glTF default of a metal
PLASTIC = {"metallicFactor": 0.0, "roughnessFactor": 0.5}

# How every renderer draws a scene unless `scene.metadata["render"]`
# says otherwise, which only holds what was changed so files don't grow
# a copy of these: read as `{**RENDER_DEFAULTS, **scene.metadata["render"]}`
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


def light_arrays(lights: list, limit: int, shadows: int) -> tuple[NDArray, list, NDArray]:
    """
    Convert `KHR_lights_punctual` lights to `shaders.LIGHT` rows with
    everything which only changes with the scene filled in, with the
    spot cone precomputed as the spec suggests so every kind of light
    is the same math in the shader.

    Parameters
    ------------
    lights : list of trimesh.scene.lighting.Light
      Directional, point and spot lights from `render_lights`.
    limit : int
      The most lights the shader has room for.
    shadows : int
      The most shadow maps, which the first directional and spot
      lights get in order: point lights never cast shadows.

    Returns
    ------------
    rows : (n,) shaders.LIGHT
      Up to `limit` lights with their color, falloff, shadow map layer
      or -1, and a position W of zero for lights infinitely far away.
    nodes : (n,) list
      Name of each light, which is its scene graph node if it has one.
    cone_angles : (n,) float
      Outer cone angle of spot lights, zero for other lights.
    """
    placed = list(lights)
    if len(placed) > limit:
        util.log.warning(f"drawing {limit} of {len(placed)} lights")
        placed = placed[:limit]
    directional = np.array(
        [isinstance(light, DirectionalLight) for light in placed], dtype=bool
    )
    spot = np.array([isinstance(light, SpotLight) for light in placed], dtype=bool)
    intensity = np.array([light.intensity for light in placed], dtype=np.float64)
    colors = np.reshape([light.color[:3] for light in placed], (-1, 3)) / 255.0
    # a missing range is NaN which `nan_to_num` makes zero: reaches forever
    radius = np.array([light.radius for light in placed], dtype=np.float64)
    inner = np.array([getattr(light, "innerConeAngle", 0.0) for light in placed])
    outer = np.array([getattr(light, "outerConeAngle", 0.0) for light in placed])
    cone_scale = np.where(
        spot, 1.0 / np.maximum(np.cos(inner) - np.cos(outer), 1e-3), 0.0
    )
    casts = directional | spot
    order = np.cumsum(casts) - 1

    rows = np.zeros(len(placed), dtype=LIGHT)
    rows["position"][:, 3] = ~directional
    rows["color"][:, :3] = colors * intensity.reshape((-1, 1))
    rows["falloff"][:, 0] = np.where(directional, 0.0, np.nan_to_num(1.0 / radius))
    rows["falloff"][:, 1] = cone_scale
    rows["falloff"][:, 2] = np.where(spot, -np.cos(outer) * cone_scale, 1.0)
    rows["shadow"][:, 0] = np.where(casts & (order < shadows), order, -1)
    return rows, [light.name for light in placed], np.where(spot, outer, 0.0)


def environment_frame(light: EnvironmentLight, transform: ArrayLike) -> NDArray:
    """
    The `shaders.FRAME` fields of an environment, in world space: the
    sky and ground colors, the up direction and the softboxes.

    Parameters
    ------------
    light : trimesh.scene.lighting.EnvironmentLight
      The environment.
    transform : (4, 4) float
      World pose of the environment's node, whose +Y is up.

    Returns
    ------------
    frame : (1,) shaders.FRAME
      With only the environment filled in.
    """
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    tint = light.color[:3] / 255.0 * light.intensity
    softboxes = light.softboxes[:MAX_SOFTBOXES]
    count = len(softboxes)
    frame = np.zeros(1, dtype=FRAME)
    frame["sky_color"][0, :3] = light.sky * tint
    frame["ground_color"][0, :3] = light.ground * tint
    frame["up_direction"][0, :3] = util.unitize(rotation[:, 1])
    frame["softbox_count"][0, 0] = count
    boxes = frame["softboxes"][0, :count]
    boxes["direction"][:, :3] = util.unitize(softboxes["direction"] @ rotation.T)
    boxes["direction"][:, 3] = softboxes["sharpness"]
    boxes["color"][:, :3] = softboxes["color"] * tint
    return frame


def geometry_to_arrays(geometry) -> GeometryArrays | None:
    """
    Convert any supported trimesh geometry into `GeometryArrays`.

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
        return points_to_arrays(geometry)
    if isinstance(geometry, Path):
        return path_to_arrays(geometry)
    util.log.warning(f"unable to draw `{type(geometry).__name__}`")
    return None


def mesh_to_arrays(mesh: Trimesh) -> GeometryArrays:
    """
    Convert a mesh to indexed triangle rows. The mesh is split along
    sharp edges with `mesh.smooth_shaded`, which keeps faces in order so
    face colors still line up with `gl_PrimitiveID`. The result is cached
    on the mesh until it or its visual changes, so viewing a mesh again
    doesn't convert it again: it is shared, so never change it.

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
    material = PBRMaterial(**PLASTIC)
    if visual.kind == "texture":
        material = visual.material.to_pbr()
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

    record, textures = material_record(material)
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


def points_to_arrays(points: PointCloud) -> GeometryArrays:
    """
    Convert a point cloud to zero-length segments, which the line
    shader draws as round dots.

    Parameters
    ------------
    points : trimesh.PointCloud
      Points to convert.

    Returns
    ------------
    arrays : GeometryArrays
      One segment per point.
    """
    vertices = util.stack_3D(points.vertices)
    rows = np.zeros(len(vertices), dtype=SEGMENT)
    rows["segment_start"] = vertices
    rows["segment_end"] = vertices
    rows["color"] = segment_colors(points.colors, len(vertices))
    return GeometryArrays(kind="points", rows=rows, material=np.zeros((), MATERIAL))


def path_to_arrays(path: Path) -> GeometryArrays:
    """
    Convert a 2D or 3D path to line segments with one color per
    segment taken from the color of the entity it came from.

    Parameters
    ------------
    path : trimesh.path.Path2D or trimesh.path.Path3D
      Path to convert.

    Returns
    ------------
    arrays : GeometryArrays
      One segment per discretized piece of every entity.
    """
    segments = path.segments
    # pad 2D paths to Z=0
    vertices = util.stack_3D(segments.reshape((-1, segments.shape[2])))
    colors = path.colors
    if colors is not None:
        colors = colors[path.segments_entity]
    rows = np.zeros(len(segments), dtype=SEGMENT)
    rows["segment_start"] = vertices[0::2]
    rows["segment_end"] = vertices[1::2]
    rows["color"] = segment_colors(colors, len(segments))
    return GeometryArrays(kind="lines", rows=rows, material=np.zeros((), MATERIAL))


def segment_colors(colors, count: int) -> NDArray:
    """
    Convert line or point colors to one RGBA row per segment.

    Parameters
    ------------
    colors : None, (4,) or (count, 3 or 4) any
      Colors in any format `to_rgba` accepts.
    count : int
      Number of segments.

    Returns
    ------------
    colors : (count, 4) uint8
      One RGBA color per segment.
    """
    if colors is None or len(colors) == 0:
        colors = DEFAULT_LINE_COLOR
    return np.broadcast_to(to_rgba(colors), (count, 4))


def material_record(material) -> tuple[NDArray, dict]:
    """
    Convert any trimesh material to the `shaders.MATERIAL` record the
    mesh shader reads, filling in glTF defaults for anything not set.

    Parameters
    ------------
    material : trimesh.visual.material.Material
      SimpleMaterial, PBRMaterial or MultiMaterial.

    Returns
    ------------
    record : () shaders.MATERIAL
      Filled in material values.
    textures : dict
      `shaders.MATERIAL_TEXTURES` name to `PIL.Image.Image` for the
      textures the material has.
    """
    pbr = material.to_pbr()
    # the glTF default for every factor which isn't set
    defaults = {
        "baseColorFactor": [255, 255, 255, 255],
        "emissiveFactor": [0.0, 0.0, 0.0],
        "metallicFactor": 1.0,
        "roughnessFactor": 1.0,
        "alphaCutoff": 0.5,
        "diffuseTransmissionFactor": 0.0,
        "diffuseTransmissionColorFactor": [1.0, 1.0, 1.0],
        "transmissionFactor": 0.0,
    }
    value = {}
    for name, default in defaults.items():
        value[name] = getattr(pbr, name)
        if value[name] is None:
            value[name] = default
    alpha_mode = ALPHA_MODES["OPAQUE"]
    if pbr.alphaMode is not None:
        alpha_mode = ALPHA_MODES[pbr.alphaMode]

    record = np.zeros((), dtype=MATERIAL)
    record["base_color"] = np.asarray(value["baseColorFactor"]) / 255.0
    record["emissive"][:3] = value["emissiveFactor"]
    record["factors"] = (
        value["metallicFactor"],
        value["roughnessFactor"],
        value["alphaCutoff"],
        value["diffuseTransmissionFactor"],
    )
    record["transmission_color"][:3] = value["diffuseTransmissionColorFactor"]
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
    return record, textures


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
    defaults, poses = autolight(scene, camera_transform)
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


def gaussian_overlap(axis_a, sharpness_a, axis_b, sharpness_b):
    """
    The integral over the sphere of the products of spherical Gaussians
    with peaks of one, written so it stays accurate when one of them is
    very sharp.

    Parameters
    ------------
    axis_a : (m, 3) float
      Unit vectors of the first Gaussians.
    sharpness_a : (m,) float
      Sharpness of the first Gaussians.
    axis_b : (3,) or (n, 3) float
      Unit vectors of the second Gaussian.
    sharpness_b : float or (n,) float
      Sharpness of the second Gaussian.

    Returns
    ------------
    overlap : (m,) or (m, n) float
      The integral for each of `axis_a` and each of `axis_b`.
    """
    axis_b = np.asarray(axis_b, dtype=np.float64)
    sharpness_b = np.asarray(sharpness_b, dtype=np.float64)
    # every first Gaussian against every second along the leading axis
    shape = (len(axis_a),) + (1,) * (axis_b.ndim - 1) + (3,)
    combined = np.linalg.norm(
        (sharpness_a[:, None] * axis_a).reshape(shape) + sharpness_b[..., None] * axis_b,
        axis=-1,
    )
    sharpness_a = sharpness_a.reshape(shape[:-1])
    total = combined + sharpness_a + sharpness_b
    exponent = 2.0 * sharpness_a * sharpness_b * (axis_a @ axis_b.T - 1.0) / total
    limited = np.maximum(combined, np.finfo(np.float64).tiny)
    return 2.0 * np.pi * np.exp(exponent) * (1.0 - np.exp(-2.0 * combined)) / limited


def environment_light(
    light: EnvironmentLight, transform: ArrayLike, direction: ArrayLike, blur
) -> NDArray[float64]:
    """
    An environment's light around a direction averaged by a spherical
    Gaussian kernel which integrates to one, which is exactly what the
    OpenGL viewer's shader evaluates.

    Parameters
    ------------
    light : trimesh.scene.lighting.EnvironmentLight
      The environment.
    transform : (4, 4) float
      World pose of the environment's node, whose +Y is up.
    direction : (3,) or (n, 3) float
      World unit vectors to look at the environment in.
    blur : float or (n,) float
      Sharpness of the averaging kernel, `DIFFUSE_SHARPNESS` for
      what a diffuse surface facing `direction` receives.

    Returns
    ------------
    radiance : (3,) or (n, 3) float
      Linear RGB radiance in each direction.
    """
    direction = util.unitize(direction)
    blur = np.asarray(blur, dtype=np.float64)
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    # the gradient is linear in height so averaging it only scales
    # the height by the kernel's mean cosine
    spread = np.exp(-2.0 * blur)
    mean_cosine = (1.0 + spread) / (1.0 - spread) - 1.0 / blur
    height = (direction @ util.unitize(rotation[:, 1])) * mean_cosine
    radiance = light.ground + (light.sky - light.ground) * (0.5 + 0.5 * height)[..., None]
    softboxes = light.softboxes
    if len(softboxes) > 0:
        kernel_peak = blur / (2.0 * np.pi * (1.0 - spread))
        towards = util.unitize(softboxes["direction"] @ rotation.T)
        overlap = gaussian_overlap(towards, softboxes["sharpness"], direction, blur)
        radiance = radiance + (overlap.T @ softboxes["color"]) * kernel_peak[..., None]
    return radiance * (light.color[:3] / 255.0 * light.intensity)


def ground_light(lights: list, transforms: ArrayLike) -> float:
    """
    How bright a white floor facing up looks with nothing shadowing it,
    which is what a shadow on the floor is darker than.

    Parameters
    ------------
    lights : list of trimesh.scene.lighting.Light
      From `render_lights`, which ends with the environment.
    transforms : (n, 4, 4) float
      World pose of each light.

    Returns
    ------------
    luminance : float
      What the floor reflects from the environment and directional lights.
    """
    transforms = np.reshape(transforms, (-1, 4, 4))
    up = util.unitize(transforms[-1, :3, 1])
    received = environment_light(lights[-1], transforms[-1], up, DIFFUSE_SHARPNESS)
    directional = [isinstance(light, DirectionalLight) for light in lights[:-1]]
    # a white surface facing a directional light reflects `C * I / pi`
    facing = np.clip(util.unitize(transforms[:-1, :3, 2]) @ up, 0.0, None) * directional
    colors = np.reshape([light.color[:3] for light in lights[:-1]], (-1, 3)) / 255.0
    intensity = np.array([light.intensity for light in lights[:-1]])
    received = received + (facing * intensity) @ colors / np.pi
    return float(received @ LUMINANCE)


def shadow_floor(scene, environment_transform: ArrayLike) -> NDArray[float64] | None:
    """
    A square floor under the lowest corner of the scene which only shows
    shadows, level with the environment and as wide as eight times the
    scene, facing up.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to put the floor under.
    environment_transform : (4, 4) float
      World pose of the environment's node, whose +Y is up and whose
      +X and +Z are the floor's edges.

    Returns
    ------------
    corners : (4, 3) float or None
      Counterclockwise seen from above, or None for an empty scene.
    """
    bounds = scene.bounds
    if bounds is None:
        return None
    right, up, back = util.unitize(np.asarray(environment_transform)[:3, :3].T)
    center = bounds.mean(axis=0)
    # a ten thousandth of the scene under it so the scene never touches it
    height = (bounds_corners(bounds) @ up).min() - 1e-4 * scene.scale
    base = center + (height - np.dot(center, up)) * up
    square = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float64)
    return base + 4.0 * scene.scale * square @ np.vstack((right, back))


def occlusion_radius(scene) -> Number:
    """
    How far ambient occlusion looks for nearby surfaces.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene being drawn.

    Returns
    ------------
    radius : float
      `OCCLUSION_RADIUS` of the scene's scale, or of one if it's empty.
    """
    if scene.bounds is None:
        return OCCLUSION_RADIUS
    return OCCLUSION_RADIUS * scene.scale
