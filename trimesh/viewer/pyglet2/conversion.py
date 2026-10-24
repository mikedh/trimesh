"""
trimesh.viewer.pyglet2.conversion
---------------------------------

Pure converters from trimesh shapes to pyglet draw objects. None of these
functions touch viewer state — they take the shared `Programs` + `Batch`
and return a `pyglet.model.Model` (or None for unrenderable input).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyglet
from pyglet.gl import (
    GL_LINES,
    GL_POINTS,
    GL_TEXTURE0,
    GL_TRIANGLES,
    glActiveTexture,
    glBindTexture,
)
from pyglet.math import Mat4
from pyglet.model import BaseMaterialGroup, Model

from ... import util
from ...typed import NDArray, float64
from ...visual import to_rgba


@dataclass(frozen=True)
class Programs:
    """The pair of shader programs every Model in this viewer draws with."""

    plain: pyglet.graphics.shader.ShaderProgram
    textured: pyglet.graphics.shader.ShaderProgram


class NodeGroup(BaseMaterialGroup):
    """
    Per-node draw group with identity equality and optional texture binding.

    Pyglet's default group equality is `(class, order, parent)`, which would
    consolidate every order-0 group into one draw call: a single `model`
    matrix would win and only one node would render. Identity equality keeps
    each node's draw distinct. `set_state` binds the texture only if there
    is one, so this single class covers both textured and untextured cases.
    """

    def __init__(self, program, texture=None, order: int = 0) -> None:
        super().__init__(material=None, program=program, order=order)
        self.texture = texture

    def set_state(self) -> None:
        if self.texture is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(self.texture.target, self.texture.id)
        self.program.use()
        self.program["model"] = self.matrix

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


def matrix_to_pyglet(matrix: NDArray[float64]) -> Mat4:
    """
    Row-major numpy 4x4 to column-major `pyglet.math.Mat4`. Skipping this
    silently transposes the upload: wrong picture, no error.
    """
    return Mat4(*np.asarray(matrix, dtype=np.float32).flatten("F"))


def to_texture(material, upsize: bool = True):
    """
    Convert a `trimesh.visual.material.Material` to a `pyglet.image.Texture`,
    or `None` if the material has no usable image. The caller must hold an
    active GL context.
    """
    image = getattr(material, "image", None)
    if image is None:
        image = getattr(material, "baseColorTexture", None)
    if image is None:
        return None
    if upsize:
        try:
            from ...visual.texture import power_resize

            image = power_resize(image)
        except BaseException:
            util.log.warning("texture power_resize failed", exc_info=True)
    with util.BytesIO() as buffer:
        image.save(buffer, format="png")
        buffer.seek(0)
        return pyglet.image.load(filename=".png", file=buffer).get_texture()


def texture_for(visual):
    """
    `(pyglet.image.Texture, uvs)` for a `TextureVisuals`, or `(None, None)`.

    trimesh stores UVs in OpenGL convention (origin bottom-left); the glTF
    loader flips V on import. Don't flip again here.
    """
    if (
        visual is None
        or getattr(visual, "uv", None) is None
        or not hasattr(visual, "material")
    ):
        return None, None
    try:
        texture = to_texture(visual.material)
    except BaseException:
        util.log.warning("failed to convert texture", exc_info=True)
        return None, None
    if texture is None:
        return None, None
    return texture, np.asarray(visual.uv, dtype=np.float32)


def build_model(
    programs: Programs,
    batch: pyglet.graphics.Batch,
    count: int,
    mode: int,
    indices,
    attributes: dict,
    *,
    texture=None,
    order: int = 0,
) -> Model:
    """Create a single-vertex-list `Model` in `batch`."""
    program = programs.textured if texture is not None else programs.plain
    group = NodeGroup(program, texture=texture, order=order)
    if indices is None:
        vertex_list = program.vertex_list(
            count, mode, batch=batch, group=group, **attributes
        )
    else:
        vertex_list = program.vertex_list_indexed(
            count, mode, indices, batch=batch, group=group, **attributes
        )
    return Model([vertex_list], [group], batch)


def to_model(geometry, programs: Programs, batch: pyglet.graphics.Batch) -> Model | None:
    """Build a `Model` for one piece of geometry, or `None` if unsupported."""
    # voxels: bake to boxes and fall through to Trimesh
    if type(geometry).__name__ == "VoxelGrid":
        geometry = geometry.as_boxes()
    kind = type(geometry).__name__

    if kind == "Trimesh":
        if getattr(geometry, "is_empty", False):
            return None
        visual = geometry.visual
        texture, uvs = texture_for(visual)
        count = len(geometry.vertices)
        colors = (
            np.ones(count * 4, dtype=np.float32)
            if texture is not None
            else (np.asarray(visual.vertex_colors, dtype=np.float32) / 255.0).ravel()
        )
        attrs = {
            "POSITION": (
                "f",
                np.asarray(geometry.vertices, dtype=np.float32).ravel(),
            ),
            "NORMAL": (
                "f",
                np.asarray(geometry.vertex_normals, dtype=np.float32).ravel(),
            ),
            "COLOR_0": ("f", colors),
        }
        if uvs is not None:
            attrs["TEXCOORD_0"] = ("f", uvs.ravel())
        return build_model(
            programs,
            batch,
            count,
            GL_TRIANGLES,
            np.asarray(geometry.faces, dtype=np.uint32).ravel(),
            attrs,
            texture=texture,
            order=1 if getattr(visual, "transparency", False) else 0,
        )

    if kind == "PointCloud":
        points = np.asarray(geometry.vertices, dtype=np.float32)
        if len(points) == 0:
            return None
        count = len(points)
        if geometry.colors is None:
            colors = np.ones(count * 4, dtype=np.float32)
        else:
            rgba = np.asarray(to_rgba(geometry.colors), dtype=np.float32) / 255.0
            # broadcast a single rgba tuple across all points if needed
            if rgba.size == 4:
                rgba = np.broadcast_to(rgba, (count, 4))
            colors = rgba.ravel()
        return build_model(
            programs,
            batch,
            count,
            GL_POINTS,
            None,
            {
                "POSITION": ("f", points.ravel()),
                "NORMAL": ("f", np.tile([0.0, 0.0, 1.0], count).astype(np.float32)),
                "COLOR_0": ("f", colors),
            },
        )

    if kind in ("Path2D", "Path3D"):
        polylines, indices, offset = [], [], 0
        for poly in geometry.discrete:
            poly = np.asarray(poly, dtype=np.float32)
            if len(poly) < 2:
                continue
            if poly.shape[1] == 2:
                poly = np.column_stack([poly, np.zeros(len(poly), dtype=np.float32)])
            polylines.append(poly)
            for i in range(len(poly) - 1):
                indices.extend((offset + i, offset + i + 1))
            offset += len(poly)
        if not polylines:
            return None
        return build_model(
            programs,
            batch,
            offset,
            GL_LINES,
            np.asarray(indices, dtype=np.uint32),
            {
                "POSITION": ("f", np.concatenate(polylines).ravel()),
                "NORMAL": ("f", np.tile([0.0, 0.0, 1.0], offset).astype(np.float32)),
                "COLOR_0": ("f", np.ones(offset * 4, dtype=np.float32)),
            },
        )

    util.log.warning(f"to_model: unsupported geometry {kind!r}")
    return None
