"""
conversion.py
---------------

Convert trimesh geometry into flat numpy arrays ready to upload
to the GPU. Nothing here imports pyglet or calls OpenGL, so all of
it can be tested anywhere.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .. import util
from ..base import Trimesh
from ..bounds import corners as bounds_corners
from ..path.path import Path
from ..points import PointCloud
from ..scene.lighting import DirectionalLight, SpotLight
from ..typed import NDArray
from ..visual.color import to_rgba
from ..voxel.base import VoxelGrid

# How a surface's alpha is used, which is glTF's `alphaMode`, and the
# integer the mesh shader compares against:
#   OPAQUE ignores alpha
#   MASK draws a pixel only where alpha is at least `alpha_cutoff`
#   BLEND draws over whatever is behind, with surfaces sorted far to near
# Clear glass from `KHR_materials_transmission` is BLEND with an alpha
# of one minus how much light passes through. Every shader writes color
# already multiplied by alpha and the framebuffer adds what is behind
# times one minus alpha, so a reflection, which isn't multiplied by
# alpha, still shows on glass which lets everything else through.
ALPHA_MODES = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}

# lines and points without colors are drawn in black
DEFAULT_LINE_COLOR = np.array([0, 0, 0, 255], dtype=np.uint8)


@dataclass
class MaterialUniforms:
    """
    A glTF metallic-roughness material with every default filled
    in, holding exactly what the mesh shader needs. The defaults
    are a matte grey plastic rather than the glTF default metal.
    """

    # linear RGBA multiplied into every pixel
    base_color: NDArray = field(default_factory=lambda: np.ones(4, dtype=np.float32))
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: NDArray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # one of the values in `ALPHA_MODES`
    alpha_mode: int = ALPHA_MODES["OPAQUE"]
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    # an invisible floor which only shows the shadows that fall on it
    shadow_catcher: bool = False
    # `KHR_materials_unlit`: the base color is the final color
    unlit: bool = False
    # the fraction of diffuse light scattered out of the other side of a
    # thin surface and its linear RGB color, `KHR_materials_diffuse_transmission`
    diffuse_transmission: float = 0.0
    diffuse_transmission_color: NDArray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )
    # `PIL.Image.Image` or None
    base_color_texture: object = None
    metallic_roughness_texture: object = None
    emissive_texture: object = None
    normal_texture: object = None
    occlusion_texture: object = None
    # the alpha channel multiplies `diffuse_transmission` and the
    # color channels multiply `diffuse_transmission_color`
    diffuse_transmission_texture: object = None
    diffuse_transmission_color_texture: object = None


@dataclass
class GeometryArrays:
    """
    Everything needed to draw one piece of geometry as numpy arrays:
    indexed triangles, or line segments where a point is a segment with
    two identical endpoints.
    """

    kind: Literal["mesh", "lines", "points"]
    # mesh: (n, 3) float32, lines and points: (n, 2, 3) float32
    vertices: NDArray
    # mesh: (n, 4) uint8 per vertex, lines and points: (n, 4) uint8 per segment
    vertex_colors: NDArray
    # mesh only: (m, 3) uint32
    faces: NDArray | None = None
    # mesh only: (n, 3) float32
    normals: NDArray | None = None
    # mesh only: (n, 2) float32
    uv: NDArray | None = None
    # mesh only: (m, 4) uint8 or None if colors are per-vertex
    face_colors: NDArray | None = None
    # mesh only: (n, 4) which joints move each vertex and how much,
    # or None for meshes without a skin
    joints: NDArray | None = None
    weights: NDArray | None = None
    material: MaterialUniforms = field(default_factory=MaterialUniforms)


@dataclass
class LightArrays:
    """
    The parts of `Scene.lights` which only change with the scene, as
    one row per light in the order the lights are drawn.
    """

    # name of each light, which is its scene graph node if it has one
    nodes: list
    # (n,) bool lights which are infinitely far away
    directional: NDArray
    # (n, 3) linear RGB times intensity in lux or candela
    colors: NDArray
    # (n, 4) `shaders.LIGHT.falloff`: one over range or zero, the spot
    # cone scale and offset which are zero and one without a cone, zero
    falloffs: NDArray
    # (n,) shadow map layer or -1 for lights which don't cast shadows
    layers: NDArray
    # (n,) outer cone angle of spot lights, zero for other lights
    cone_angles: NDArray


def light_arrays(lights: list, limit: int, shadows: int) -> LightArrays:
    """
    Convert `KHR_lights_punctual` lights to arrays, with the spot cone
    precomputed as the spec suggests so every kind of light is the same
    math in the shader.

    Parameters
    ------------
    lights : list of trimesh.scene.lighting.Light
      Lights from `trimesh.scene.metadata.Studio.lights`.
    limit : int
      The most lights the shader has room for.
    shadows : int
      The most shadow maps, which the first directional and spot
      lights get in order: point lights never cast shadows.

    Returns
    ------------
    arrays : LightArrays
      One row per light, up to `limit`.
    """
    placed = list(lights)
    if len(placed) > limit:
        util.log.debug(f"drawing {limit} of {len(placed)} lights")
        placed = placed[:limit]
    directional = np.array(
        [isinstance(light, DirectionalLight) for light in placed], dtype=bool
    )
    spot = np.array([isinstance(light, SpotLight) for light in placed], dtype=bool)
    intensity = np.array([light.intensity for light in placed], dtype=np.float64)
    colors = np.reshape([light.color[:3] for light in placed], (-1, 3)) / 255.0
    # a missing range is NaN which `nan_to_num` makes zero: reaches forever
    radius = np.array([light.radius for light in placed], dtype=np.float64)
    inverse_range = np.where(directional, 0.0, np.nan_to_num(1.0 / radius))
    inner = np.array([getattr(light, "innerConeAngle", 0.0) for light in placed])
    outer = np.array([getattr(light, "outerConeAngle", 0.0) for light in placed])
    cone_scale = np.where(
        spot, 1.0 / np.maximum(np.cos(inner) - np.cos(outer), 1e-3), 0.0
    )
    cone_offset = np.where(spot, -np.cos(outer) * cone_scale, 1.0)
    casts = directional | spot
    order = np.cumsum(casts) - 1
    layers = np.where(casts & (order < shadows), order, -1)
    return LightArrays(
        nodes=[light.name for light in placed],
        directional=directional,
        colors=colors * intensity.reshape((-1, 1)),
        falloffs=np.column_stack(
            (inverse_range, cone_scale, cone_offset, np.zeros(len(placed)))
        ).reshape((-1, 4)),
        layers=layers,
        cone_angles=np.where(spot, outer, 0.0),
    )


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
    util.log.debug(f"unable to draw `{type(geometry).__name__}`")
    return None


def mesh_to_arrays(mesh: Trimesh) -> GeometryArrays:
    """
    Convert a mesh to indexed triangle arrays. The mesh is split
    along sharp edges with `mesh.smooth_shaded`, which keeps faces in
    order so face colors still line up with `gl_PrimitiveID`.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Mesh to convert.

    Returns
    ------------
    arrays : GeometryArrays
      Indexed triangles with per-vertex normals, uv and colors.
    """
    smooth = mesh.smooth_shaded
    visual = smooth.visual
    count = len(smooth.vertices)

    # every mesh gets every attribute so the shader never branches:
    # the defaults are white colors and zero uv
    vertex_colors = np.full((count, 4), 255, dtype=np.uint8)
    uv = np.zeros((count, 2), dtype=np.float32)
    face_colors = None
    material = MaterialUniforms()

    if visual.kind == "texture":
        material = material_to_uniforms(visual.material)
        if visual.uv is not None and len(visual.uv) == count:
            uv = np.asarray(visual.uv[:, :2], dtype=np.float32)
        # per-vertex colors multiply the material like glTF `COLOR_0`
        colors = visual.vertex_attributes.get("color")
        if colors is not None and len(colors) == count:
            vertex_colors = np.asarray(to_rgba(colors), dtype=np.uint8)
            material.alpha_mode = max(
                material.alpha_mode, transparency_mode(vertex_colors)
            )
    elif visual.kind == "face":
        face_colors = np.asarray(visual.face_colors, dtype=np.uint8)
        material.alpha_mode = transparency_mode(face_colors)
    else:
        # vertex colors, or the default color if nothing is set
        vertex_colors = np.asarray(visual.vertex_colors, dtype=np.uint8)
        material.alpha_mode = transparency_mode(vertex_colors)

    # the joints and weights of a skinned mesh, which `smooth_shaded` keeps
    joints = smooth.vertex_attributes.get("joints")
    weights = smooth.vertex_attributes.get("weights")
    if joints is None or weights is None:
        joints, weights = None, None

    return GeometryArrays(
        kind="mesh",
        joints=joints,
        weights=weights,
        vertices=np.asarray(smooth.vertices, dtype=np.float32),
        faces=np.asarray(smooth.faces, dtype=np.uint32),
        normals=np.asarray(smooth.vertex_normals, dtype=np.float32),
        uv=uv,
        vertex_colors=vertex_colors,
        face_colors=face_colors,
        material=material,
    )


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
    vertices = np.asarray(util.stack_3D(points.vertices), dtype=np.float32)
    colors = segment_colors(points.colors, len(vertices))
    return GeometryArrays(
        kind="points",
        vertices=np.repeat(vertices[:, None, :], 2, axis=1),
        vertex_colors=colors,
    )


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
    return GeometryArrays(
        kind="lines",
        vertices=np.asarray(vertices, dtype=np.float32).reshape((-1, 2, 3)),
        vertex_colors=segment_colors(colors, len(segments)),
    )


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
    return np.ascontiguousarray(
        np.broadcast_to(to_rgba(colors), (count, 4)), dtype=np.uint8
    )


def transparency_mode(colors: NDArray) -> int:
    """
    Pick the alpha mode for uint8 RGBA colors.

    Parameters
    ------------
    colors : (n, 4) uint8
      RGBA colors.

    Returns
    ------------
    mode : int
      `ALPHA_MODES["BLEND"]` if any color is translucent.
    """
    if (colors[:, 3] < 255).any():
        return ALPHA_MODES["BLEND"]
    return ALPHA_MODES["OPAQUE"]


def material_to_uniforms(material) -> MaterialUniforms:
    """
    Convert any trimesh material to the values the mesh shader
    needs, filling in glTF defaults for anything not set.

    Parameters
    ------------
    material : trimesh.visual.material.Material
      SimpleMaterial, PBRMaterial or MultiMaterial.

    Returns
    ------------
    uniforms : MaterialUniforms
      Filled in material values.
    """
    pbr = material.to_pbr()
    uniforms = MaterialUniforms(
        metallic=1.0,
        roughness=1.0,
        double_sided=bool(pbr.doubleSided),
        unlit=bool(pbr.unlit),
        base_color_texture=pbr.baseColorTexture,
        metallic_roughness_texture=pbr.metallicRoughnessTexture,
        emissive_texture=pbr.emissiveTexture,
        normal_texture=pbr.normalTexture,
        occlusion_texture=pbr.occlusionTexture,
        diffuse_transmission_texture=pbr.diffuseTransmissionTexture,
        diffuse_transmission_color_texture=pbr.diffuseTransmissionColorTexture,
    )
    if pbr.baseColorFactor is not None:
        uniforms.base_color = np.asarray(pbr.baseColorFactor, dtype=np.float32) / 255.0
    if pbr.metallicFactor is not None:
        uniforms.metallic = float(pbr.metallicFactor)
    if pbr.roughnessFactor is not None:
        uniforms.roughness = float(pbr.roughnessFactor)
    if pbr.emissiveFactor is not None:
        uniforms.emissive = np.asarray(pbr.emissiveFactor, dtype=np.float32)
    if pbr.alphaMode is not None:
        uniforms.alpha_mode = ALPHA_MODES[pbr.alphaMode]
    if pbr.alphaCutoff is not None:
        uniforms.alpha_cutoff = float(pbr.alphaCutoff)
    if pbr.diffuseTransmissionFactor is not None:
        uniforms.diffuse_transmission = float(pbr.diffuseTransmissionFactor)
    if pbr.diffuseTransmissionColorFactor is not None:
        uniforms.diffuse_transmission_color = np.asarray(
            pbr.diffuseTransmissionColorFactor, dtype=np.float32
        )
    if pbr.transmissionFactor is not None:
        # glass, which is BLEND as `ALPHA_MODES` describes
        uniforms.base_color[3] *= 1.0 - pbr.transmissionFactor
        uniforms.alpha_mode = ALPHA_MODES["BLEND"]
    return uniforms


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
    count = len(world)
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

    orthographic = np.zeros((count, 4, 4))
    near = depth.min(axis=1) - margin
    orthographic[:, [0, 1], [0, 1]] = 2.0 / size
    orthographic[:, :2, 3] = -(high + low) / size
    orthographic[:, 2, 2] = -2.0 / (far - near)
    orthographic[:, 2, 3] = -(far + near) / (far - near)
    orthographic[:, 3, 3] = 1.0

    # a spot light can't see behind itself so its near plane is in front,
    # where a 24 bit depth buffer tells apart `far^2 / (near * 2^24)` at
    # the far plane, so the near plane is kept where that is under 0.01%
    perspective = np.zeros((count, 4, 4))
    near = np.maximum(depth.min(axis=1), far * 1e-3)
    # a cone between straight ahead and a hemisphere has a finite focal length
    focal = 1.0 / np.tan(np.clip(cone_angles, 1e-3, np.pi / 2.0 - 1e-3))
    perspective[:, 0, 0] = focal
    perspective[:, 1, 1] = focal
    perspective[:, 2, 2] = (far + near) / (near - far)
    perspective[:, 2, 3] = 2.0 * far * near / (near - far)
    perspective[:, 3, 2] = -1.0

    projection = np.where(directional.reshape((-1, 1, 1)), orthographic, perspective)
    texels = np.column_stack(
        (
            np.where(directional, size.max(axis=1), 0.0),
            np.where(directional, 0.0, 2.0 / focal),
        )
    ) / float(resolution)
    return np.matmul(projection, to_light), texels
