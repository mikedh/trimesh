"""
conversion.py
---------------

Convert geometry and lights to the structured arrays in `opengl/shaders`.
Nothing here imports pyglet or calls OpenGL.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .. import util
from ..base import Trimesh
from ..path.path import Path
from ..points import PointCloud
from ..scene.lighting import (
    EnvironmentLight,
    autolight,
)
from ..typed import ArrayLike, NDArray, float64
from ..visual.color import to_rgba
from ..visual.material import PBRMaterial
from ..voxel.base import VoxelGrid
from .opengl.shaders import (
    ALPHA_MODES,
    MATERIAL,
    MATERIAL_TEXTURES,
    SEGMENT,
    SKIN,
    VERTEX,
)

# render settings, overridden by keywords and never stored in the scene
RENDER_DEFAULTS = {
    # an invisible floor which only shows shadows
    "ground_shadow": True,
    # draw the environment as the background
    "show_environment": False,
    # pixels across lines and points
    "line_width": 2.0,
    "point_size": 6.0,
    # multiplies light before tone mapping, but not lines, points or unlit
    "exposure": 1.0,
    # glare around bright pixels, drawn by the OpenGL viewer with quality
    "bloom": 1.0,
}

# ambient occlusion radius as a fraction of `scene.scale`
OCCLUSION_RADIUS = 0.1


@dataclass
class GeometryArrays:
    """
    GPU arrays for one geometry, where points are zero length segments.
    """

    kind: Literal["mesh", "lines", "points"]
    # `shaders.VERTEX` rows for a mesh, `shaders.SEGMENT` rows otherwise
    rows: NDArray
    # one `shaders.MATERIAL` record
    material: NDArray
    # mesh only: (m, 3) uint32
    faces: NDArray | None = None
    # mesh only: `shaders.SKIN` rows or None
    skin: NDArray | None = None
    # mesh only: (m, 4) uint8 or None if colors are per-vertex
    face_colors: NDArray | None = None
    # `shaders.MATERIAL_TEXTURES` name to `PIL.Image.Image`
    textures: dict = field(default_factory=dict)
    # rows are stored relative to this, as float32 can't hold far coordinates
    origin: NDArray = field(default_factory=lambda: np.zeros(3))


def geometry_to_arrays(geometry) -> GeometryArrays | None:
    """
    Convert geometry to `GeometryArrays`, cached on meshes and read-only.

    Parameters
    ------------
    geometry : Trimesh, PointCloud, Path2D, Path3D or VoxelGrid
      Geometry to convert.

    Returns
    ------------
    arrays : GeometryArrays or None
      None if empty or unsupported.
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
        origin = (
            np.minimum(start, end).min(axis=0) + np.maximum(start, end).max(axis=0)
        ) / 2
        rows = np.zeros(len(start), dtype=SEGMENT)
        rows["segment_start"], rows["segment_end"] = start - origin, end - origin
        # uncolored lines and points are black
        rows["color"] = (0, 0, 0, 255)
        if colors is not None and len(colors) > 0:
            rows["color"] = to_rgba(colors)
        return GeometryArrays(
            kind=kind, rows=rows, material=np.zeros((), MATERIAL), origin=origin
        )

    # keyed by material too as it isn't in the visual hash; the mesh
    # cache is cleared when vertices or faces change
    material = getattr(geometry.visual, "material", None)
    key = f"viewer_arrays_{hash(geometry.visual)}_{hash(material)}"
    if geometry._cache[key] is not None:
        return geometry._cache[key]
    smooth = geometry.smooth_shaded
    visual = smooth.visual
    # every attribute is set so the shader never branches
    rows = np.zeros(len(smooth.vertices), dtype=VERTEX)
    # `smooth_shaded` keeps joints and weights
    skin, attributes = None, smooth.vertex_attributes
    origin = smooth.bounds.mean(axis=0)
    if attributes.get("joints") is not None and attributes.get("weights") is not None:
        skin = np.zeros(len(rows), dtype=SKIN)
        skin["joints"], skin["weights"] = attributes["joints"], attributes["weights"]
        # skinned vertices are in bind space, which joints move as they are
        origin = np.zeros(3)
    rows["position"] = smooth.vertices - origin
    rows["normal"] = smooth.vertex_normals
    rows["color"] = 255
    # default to matte plastic rather than glTF's metal
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
        rows["color"] = visual.vertex_colors

    # glTF defaults for factors not set
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
    textures = {}
    for name, (attribute, _) in MATERIAL_TEXTURES.items():
        image = getattr(pbr, attribute)
        if image is not None:
            textures[name] = image
    # transmission and translucent colors blend whatever the material says
    alpha = rows["color"][:, 3]
    if face_colors is not None:
        alpha = face_colors[:, 3]
    if pbr.transmissionFactor is not None or (alpha < 255).any():
        record["flags"][0] = max(record["flags"][0], ALPHA_MODES["BLEND"])
    record["has_textures"][:2] = ("normal_texture" in textures, face_colors is not None)

    arrays = GeometryArrays(
        kind="mesh",
        rows=rows,
        material=record,
        faces=np.asarray(smooth.faces, dtype=np.uint32),
        skin=skin,
        face_colors=face_colors,
        textures=textures,
        origin=origin,
    )
    geometry._cache[key] = arrays
    return arrays


def render_lights(scene, camera_transform: ArrayLike) -> tuple[list, NDArray[float64]]:
    """
    Lights a render uses: the scene's own, or `autolight`, ending with one environment.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to light, not changed.
    camera_transform : (4, 4) float
      Initial camera pose.

    Returns
    ------------
    lights : list of trimesh.scene.lighting.Light
      Punctual lights then one `EnvironmentLight`.
    transforms : (n, 4, 4) float
      World pose of each light.
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
    # the first environment, or the studio's
    pairs = [pair for pair in pairs if not isinstance(pair[0], EnvironmentLight)]
    lights, transforms = zip(*pairs, [*environments, studio[-1]][0])
    return list(lights), np.reshape(transforms, (-1, 4, 4))
