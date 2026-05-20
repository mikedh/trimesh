"""
trimesh.viewer.pyglet2.helper
-----------------------------

Small types, constants, and free functions shared by `window` and
`conversion`. Nothing here depends on a live GL context except the
shader source strings, which are only consumed by `ctx.create_program`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pyglet.math import Mat4

from ... import util
from ..trackball import Trackball

# (input, action). Mouse / drag bindings handled directly by `on_mouse_*`.
MOUSE: tuple[tuple[str, str], ...] = (
    ("click + drag", "rotate"),
    ("ctrl + drag", "pan"),
    ("shift + drag", "roll"),
    ("wheel", "zoom"),
    ("arrows", "nudge"),
)

# (key, action, SceneViewer method). Single source of truth for both
# `on_key_press` dispatch and the help message.
KEYS: tuple[tuple[str, str, str], ...] = (
    ("w", "toggle wireframe", "toggle_wireframe"),
    ("z", "reset view", "reset_view"),
    ("c", "toggle backface culling", "toggle_culling"),
    ("a", "toggle axis", "toggle_axis"),
    ("g", "toggle grid", "toggle_grid"),
    ("f", "toggle fullscreen", "toggle_fullscreen"),
    ("m", "maximize", "maximize"),
    ("q", "quit", "close"),
    ("h", "help", "print_help"),
)


# Possible values for `ViewerState.axis`. Only "world" is wired up in
# pyglet2 today; "all" and "without_world" are pre-declared to match
# the legacy pyglet1 viewer's cycle.
AxisMode = Literal["world", "all", "without_world"]


@dataclass
class ViewerState:
    """All viewer UI state, in one place. None of these fields are read by
    pyglet, so adding a new toggle is just adding a field here + a method."""

    trackball: Trackball
    wireframe: bool = False
    cull: bool = True
    fullscreen: bool = False
    axis: AxisMode | None = None
    grid: bool = False
    hidden: set = field(default_factory=set)
    fixed: set = field(default_factory=set)


def build_help() -> str:
    """Render the help text. Column width is derived from the data so
    adding a binding never breaks alignment."""
    rows = (*MOUSE, *((k, action) for k, action, _ in KEYS))
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        (
            "trimesh SceneViewer",
            "",
            *(f"  {label:<{width}}  {action}" for label, action in rows),
        )
    )


def drop_model(model) -> None:
    """Best-effort delete of every `VertexList` owned by a `pyglet.model.Model`."""
    if model is None:
        return
    for vertex_list in getattr(model, "vertex_lists", ()):
        try:
            vertex_list.delete()
        except BaseException:
            pass


def ensure_decoration(current, flag, factory):
    """
    Reconcile a decoration model (axis, grid, ...) against a flag.

    - `flag` truthy + `current` is None: build via `factory()` and return it.
    - `flag` falsy + `current` is not None: drop it and return None.
    - Otherwise: return `current` unchanged.

    Failures in `factory` are logged and treated as "no decoration".
    """
    if flag and current is None:
        try:
            return factory()
        except BaseException:
            util.log.warning("decoration factory failed", exc_info=True)
            return None
    if not flag and current is not None:
        drop_model(current)
        return None
    return current


# A literal zero matrix produces NaN clip coords on Mesa-EGL and drops the
# whole draw, so collapse to a vanishingly small scale instead.
_TINY = 1e-30
HIDDEN_MATRIX = Mat4(_TINY, 0, 0, 0, 0, _TINY, 0, 0, 0, 0, _TINY, 0, 0, 0, 0, 1)


# Custom fragment shaders. Pyglet's stock fragment shader does
# `final_colors = color_0 * l * 0.75 + color_0 * vec4(0.25)`, which scales
# the alpha channel by the Lambert factor `l` and makes opaque meshes blend
# through to the background under `glEnable(GL_BLEND)`. These variants apply
# the same Lambert math to RGB but pass the input alpha through unchanged.
FRAG_SRC = """#version 330 core
    in vec4 color_0;
    in vec3 normal;
    in vec3 position;
    out vec4 final_colors;

    void main()
    {
        float l = dot(normalize(-position), normalize(normal));
        vec3 lit = (color_0 * l * 0.75 + color_0 * 0.25).rgb;
        final_colors = vec4(lit, color_0.a);
    }
"""

TEXTURED_FRAG_SRC = """#version 330 core
    in vec4 color_0;
    in vec3 normal;
    in vec2 texcoord_0;
    in vec3 position;
    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main()
    {
        float l = dot(normalize(-position), normalize(normal));
        vec4 tex_color = texture(our_texture, texcoord_0) * color_0;
        vec3 lit = (tex_color * l * 0.75 + tex_color * 0.25).rgb;
        final_colors = vec4(lit, tex_color.a);
    }
"""
