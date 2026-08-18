"""
gltf_extensions.py
------------------

Extension registry for glTF import/export with scope-based handlers.
Each scope has a TypedDict defining the context passed to handlers.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias, TypedDict

import numpy as np

from ...constants import log
from ...iteration import IndexedDict
from ...typed import NDArray

# Scopes define where in the glTF load/export process handlers run:
#   material            - after parsing material, can override PBR values
#   texture_source      - when resolving texture image index
#   primitive           - after loading primitive, can add face_attributes
#   primitive_preprocess - before accessor reads, can modify accessors in-place
#   primitive_export    - during mesh export, can compress/modify primitive data
#   scene               - after parsing a header, can return scene-level objects
#   scene_export        - during export, can write document and node extensions
Scope: TypeAlias = Literal[
    "material",
    "texture_source",
    "primitive",
    "primitive_preprocess",
    "primitive_export",
    "scene",
    "scene_export",
]


# ----------------------------------------------------------------------
# TypedDict contexts for each scope
# ----------------------------------------------------------------------
#
# These TypedDicts define the MINIMUM fields passed to handlers for each scope.
# Additional fields may be added in future versions for new functionality.
#
# FOR FORWARD COMPATIBILITY: Handlers should access only the fields they need
# and ignore unknown fields. The context is passed as a plain dict at runtime,
# so handlers can safely use dict.get() for optional access or simply not
# reference fields they don't need.
#
# Example handler pattern:
#
#     def my_handler(context: MaterialContext) -> dict | None:
#         # Access only what you need - additional fields won't break this
#         data = context["data"]
#         images = context["images"]
#         return {"baseColorFactor": [1, 0, 0, 1]}
#
# ----------------------------------------------------------------------


class MaterialContext(TypedDict):
    """Context for material scope handlers."""

    data: dict[str, Any]
    parse_textures: Callable[..., dict[str, Any]]
    images: list


class TextureSourceContext(TypedDict):
    """Context for texture_source scope handlers."""

    data: dict[str, Any]


class PrimitiveContext(TypedDict):
    """Context for primitive scope handlers (post-load)."""

    data: dict[str, Any]
    primitive: dict
    mesh_kwargs: dict
    accessors: list


class PrimitivePreprocessContext(TypedDict):
    """Context for primitive_preprocess scope handlers (pre-load)."""

    data: dict[str, Any]
    primitive: dict
    accessors: list
    views: list


class PrimitiveExportContext(TypedDict):
    """Context for primitive_export scope handlers (during export)."""

    mesh: Any
    name: str
    tree: dict
    # a `bufferView` is the position of an entry in here, so the order matters
    buffer_items: IndexedDict
    primitive: dict
    # the arrays the primitive's accessors were built from, in the dtype they
    # would have been written with, keyed by accessor index: a handler storing
    # them itself never unpacks bytes back into numpy, and the exporter stores
    # them after all if no handler claims them
    arrays: dict[int, NDArray]


class SceneContext(TypedDict):
    """Context for scene scope handlers (post-load)."""

    data: dict[str, Any]
    header: dict
    # {node index in the file : node name in the scene}
    names: dict[int, str]


class SceneExportContext(TypedDict):
    """Context for scene_export scope handlers (during export)."""

    scene: Any
    # GLTF header, mutated in place, with `nodes` already populated
    tree: dict
    # {node name : index in `tree["nodes"]`}
    node_index: dict[str, int]


# Handler type alias - handlers receive a context dict
Handler: TypeAlias = Callable[[Any], Any]

# callback to parse material dict and resolve texture references
# signature: (*, data: dict) -> dict
ParseTextures: TypeAlias = Callable[..., dict[str, Any]]

# the context each scope hands its handlers, which is also the runtime source of
# truth for which scopes exist and what a caller has to pass: a `TypedDict` knows
# its own `__required_keys__` so neither has to be repeated in a docstring
CONTEXT: dict[Scope, type] = {
    "material": MaterialContext,
    "texture_source": TextureSourceContext,
    "primitive": PrimitiveContext,
    "primitive_preprocess": PrimitivePreprocessContext,
    "primitive_export": PrimitiveExportContext,
    "scene": SceneContext,
    "scene_export": SceneExportContext,
}

# Registry: {scope: {extension_name: handler}}
_handlers: dict[Scope, dict[str, Handler]] = {}


def _deep_merge(target: dict, source: dict) -> None:
    """
    Recursively merge source dict into target dict.

    Parameters
    ----------
    target
      Dict to merge into (modified in place)
    source
      Dict to merge from
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            # Both are dicts - recurse
            _deep_merge(target[key], value)
        else:
            # Overwrite or set new key
            target[key] = value


def register_handler(name: str, scope: Scope) -> Callable[[Handler], Handler]:
    """
    Decorator to register a handler for a glTF extension.

    Parameters
    ----------
    name
      Extension name, e.g. "KHR_materials_pbrSpecularGlossiness".
    scope
      Handler scope, e.g. "material", "texture_source", "primitive".

    Returns
    -------
    decorator
      Function that registers the handler and returns it unchanged.

    Example
    -------
    >>> @register_handler("MY_extension", scope="material")
    ... def my_handler(context: MaterialContext) -> dict | None:
    ...     data = context["data"]
    ...     images = context["images"]
    ...     return {"baseColorFactor": [1, 0, 0, 1]}
    """
    if scope not in CONTEXT:
        # a misspelled scope would otherwise register a handler
        # nothing ever dispatches, which is silent and permanent
        raise ValueError(f"`{scope}` is not a scope: {sorted(CONTEXT)}")

    if scope not in _handlers:
        _handlers[scope] = {}

    def decorator(func: Handler) -> Handler:
        if name in _handlers[scope]:
            # replacing a built-in is a fair thing to want from a
            # registry, doing it by accident is not
            log.warning(f"`{name}` already registered for `{scope}`, replacing!")
        _handlers[scope][name] = func
        return func

    return decorator


def registered(scope: Scope) -> set:
    """
    Extension names which have a handler registered for a scope.

    Parameters
    ----------
    scope
      Handler scope to look up.

    Returns
    -------
    names
      Extension names with a handler for the scope.
    """
    return set(_handlers.get(scope, {}).keys())


def unregistered(extensions: Iterable[str], scope: Scope) -> set:
    """
    Find extension names with no registered handler for a scope.

    Parameters
    ----------
    extensions
      Extension names, i.e. the keys of a glTF "extensions" dict.
    scope
      Handler scope to check against.

    Returns
    -------
    missing
      Extension names with no handler registered for the scope.
    """
    return set(extensions) - _handlers.get(scope, {}).keys()


def handle_extensions(
    *,
    extensions: dict[str, Any] | None,
    scope: Scope,
    **kwargs,
) -> dict[str, Any]:
    """
    Run the handlers for the extensions found on a glTF element.

    Parameters
    ----------
    extensions
      The "extensions" dict from a glTF element, or None.
    scope
      Handler scope to invoke.
    **kwargs
      The rest of the scope's context, i.e. everything `CONTEXT[scope]`
      requires except `data`, which is filled in per extension here.

    Returns
    -------
    results
      {extension name : handler result} for handlers which returned
      something. What to do with them is the caller's business: this
      runs handlers and nothing else.
    """
    handlers = _handlers.get(scope)
    if not extensions or not handlers:
        return {}

    results = {}
    # one context reused across the loop as only `data` varies
    context = {"data": None, **kwargs}
    for name, data in extensions.items():
        handler = handlers.get(name)
        if handler is None:
            continue
        context["data"] = data
        try:
            if (result := handler(context)) is not None:
                results[name] = result
        except Exception as e:
            _blame(scope, context, name, e)

    return results


def _blame(scope: Scope, context: dict, name: str, error: Exception) -> None:
    """
    Decide whether a handler blowing up is a bad file or a caller bug.

    Only reached once something has already failed, which is what keeps
    the context check off the dispatch path entirely.

    Parameters
    ----------
    scope
      Scope the handler was registered for.
    context
      What the handler was actually passed.
    name
      Extension name, for the message.
    error
      What the handler raised.

    Raises
    ------
    ValueError
      If the caller never passed part of the scope's context.
    """
    missing = CONTEXT[scope].__required_keys__ - context.keys()
    if missing:
        raise ValueError(f"`{scope}` handlers need {sorted(missing)}!") from error

    log.warning(f"failed to process extension {name}", exc_info=True)


def merge_results(results: dict, target: dict) -> None:
    """
    Merge handler results into a target dict, in place.

    Parameters
    ----------
    results
      What `handle_extensions` returned.
    target
      Dict to merge into, i.e. the kwargs a geometry is built from.
    """
    for result in results.values():
        if not isinstance(result, dict):
            continue
        for key, value in result.items():
            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                _deep_merge(target[key], value)
            else:
                target[key] = value


def export_extensions(*, scope: Scope, **kwargs) -> dict:
    """
    Run every handler registered for an export scope.

    Unlike `handle_extensions` there is nothing in the tree to trigger on,
    as the handlers are what put it there, so the registration list is the
    work list.

    Parameters
    ----------
    scope
      Handler scope to invoke, i.e. "primitive_export" or "scene_export".
    **kwargs
      The scope's context, i.e. everything `CONTEXT[scope]` requires.

    Returns
    -------
    results
      {extension name : handler result} for handlers which returned
      something, i.e. which claimed the thing they were offered.
    """
    results = {}
    for name, handler in _handlers.get(scope, {}).items():
        try:
            if (result := handler(kwargs)) is not None:
                results[name] = result
        except Exception as e:
            _blame(scope, kwargs, name, e)

    return results


# ----------------------------------------------------------------------
# Built-in handlers
# ----------------------------------------------------------------------


@register_handler("KHR_materials_pbrSpecularGlossiness", scope="material")
def _specular_glossiness(context: MaterialContext) -> dict[str, Any] | None:
    """
    Convert specular-glossiness material to PBR metallic-roughness.

    Parameters
    ----------
    context
      MaterialContext with extension data, parse_textures function, and images.

    Returns
    -------
    pbr_dict
      PBR metallic-roughness parameters, or None on failure.
    """
    try:
        from ...visual.gloss import specular_to_pbr

        return specular_to_pbr(**context["parse_textures"](data=context["data"]))
    except Exception:
        log.debug("failed to convert specular-glossiness", exc_info=True)
        return None


@dataclass
class GltfLights:
    """
    The `KHR_lights_punctual` extension.

    A file stores its lights once in a document-level array and has each
    node reference one by index, so both halves are written and read here
    together rather than from opposite ends of the exporter. Every light
    is named after the node carrying it, which is what lets `Scene.lights`
    and the scene graph refer to the same thing.
    """

    # the extension key, and trimesh light class name : GLTF type with the
    # reverse for loading. `ClassVar` is what keeps a dataclass from turning
    # these into fields, so they stay lookups on the class
    NAME: ClassVar[str] = "KHR_lights_punctual"
    TYPES: ClassVar[dict] = {
        "DirectionalLight": "directional",
        "PointLight": "point",
        "SpotLight": "spot",
    }
    CLASSES: ClassVar[dict] = {v: k for k, v in TYPES.items()}

    lights: list = field(default_factory=list)

    @classmethod
    def from_gltf(cls, header: dict, names: dict) -> "GltfLights":
        """
        Collect the lights a header defines, named after their nodes.

        Parameters
        ----------
        header
          Parsed GLTF header.
        names
          {node index in the file : node name in the scene}

        Returns
        -------
        lights
          One light per node reference, in file order.
        """
        from ...scene import lighting
        from ...visual import color

        stored = header.get("extensions", {}).get(cls.NAME, {}).get("lights", [])

        lights = []
        # walk the nodes rather than the array: a light is only in the scene
        # if a node carries it, and two nodes may share one array entry
        for index, node in enumerate(header.get("nodes", [])):
            reference = node.get("extensions", {}).get(cls.NAME, {}).get("light")
            if reference is None:
                continue
            if not isinstance(reference, int) or reference >= len(stored):
                log.warning("node references a missing light!")
                continue

            entry = stored[reference]
            constructor = getattr(lighting, cls.CLASSES.get(entry.get("type"), ""), None)
            if constructor is None:
                log.warning(f"unsupported light type `{entry.get('type')}`!")
                continue

            kwargs = {
                # name it after the node so `Scene.lights` and the graph
                # agree, which is what a re-export needs
                "name": names[index],
                "intensity": entry.get("intensity", 1.0),
                "radius": entry.get("range"),
            }
            if "color" in entry:
                kwargs["color"] = color.to_rgba(
                    np.array(entry["color"], dtype=np.float64)
                )
            if constructor is lighting.SpotLight:
                spot = entry.get("spot", {})
                # outer first as the inner setter validates against it
                kwargs["outerConeAngle"] = spot.get("outerConeAngle", np.pi / 4.0)
                kwargs["innerConeAngle"] = spot.get("innerConeAngle", 0.0)

            lights.append(constructor(**kwargs))

        return cls(lights=lights)

    def to_gltf(self, tree: dict, node_index: dict) -> None:
        """
        Write the light array and every node reference into a tree.

        Parameters
        ----------
        tree
          GLTF header, mutated in place.
        node_index
          {node name : index in `tree["nodes"]`}
        """
        from ...visual import color

        stored = []
        for light in self.lights:
            entry = {
                "name": light.name,
                "type": self.TYPES[type(light).__name__],
                # the extension stores linear RGB in the 0.0 - 1.0 range
                "color": color.to_float(light.color)[:3].tolist(),
                "intensity": float(light.intensity),
            }
            if light.radius is not None:
                entry["range"] = float(light.radius)
            if entry["type"] == "spot":
                # only a spot has a cone, and the extension defaults differ
                # from ours so always write both rather than guessing
                entry["spot"] = {
                    "innerConeAngle": float(light.innerConeAngle),
                    "outerConeAngle": float(light.outerConeAngle),
                }
            stored.append(entry)

        tree["extensions"] = {**tree.get("extensions", {}), self.NAME: {"lights": stored}}

        # a node refers to a light by its position in the array above, which
        # is why both are written here rather than from separate places
        for i, light in enumerate(self.lights):
            index = node_index.get(light.name)
            if index is None:
                log.warning(f"light `{light.name}` has no node!")
                continue
            node = tree["nodes"][index]
            node["extensions"] = {**node.get("extensions", {}), self.NAME: {"light": i}}


@register_handler(GltfLights.NAME, scope="scene")
def _lights_load(context: SceneContext) -> list | None:
    """
    Resolve every node's light reference against the document array.

    Parameters
    ----------
    context
      SceneContext with the parsed header and resolved node names.

    Returns
    -------
    lights
      Trimesh lights, or None if the file defined none.
    """
    lights = GltfLights.from_gltf(context["header"], context["names"]).lights
    return lights if len(lights) > 0 else None


@register_handler(GltfLights.NAME, scope="scene_export")
def _lights_export(context: SceneExportContext) -> bool | None:
    """
    Write the document light array and each node's reference to one.

    Parameters
    ----------
    context
      SceneExportContext, whose `tree` we mutate.

    Returns
    -------
    written
      True if lights were stored, None if the scene had none.
    """
    scene = context["scene"]
    # note this deliberately checks `has_lights` rather than `lights`,
    # which would generate a default pair for a scene that never had any
    if not scene.has_lights:
        return None

    GltfLights(scene.lights).to_gltf(context["tree"], context["node_index"])
    return True


@register_handler("EXT_texture_webp", scope="texture_source")
def _texture_webp_source(context: TextureSourceContext) -> int | None:
    """
    Return image source index from EXT_texture_webp.

    Parameters
    ----------
    context
      TextureSourceContext with extension data.

    Returns
    -------
    source_index
      Index into glTF images array, or None if not present.
    """
    return context["data"].get("source")


# the optional attributes draco can absorb, as
# (glTF name, `DracoPy.encode` keyword, `DracoPy.AttributeType`)
_draco_optional = (
    ("COLOR_0", "colors", "COLOR"),
    ("TEXCOORD_0", "tex_coord", "TEX_COORD"),
    ("NORMAL", "normals", "NORMAL"),
)

# how hard draco tries, which trades export time for size and is not lossy
_DRACO_COMPRESSION = 6

# bits draco quantizes positions onto, which is what blender emits: a vertex
# moves up to half a step, i.e. `mesh.extents.max() * 2**-(bits + 1)`
_DRACO_QUANTIZATION = 14


@register_handler("KHR_draco_mesh_compression", scope="primitive_preprocess")
def _draco_decode(context: PrimitivePreprocessContext) -> None:
    """
    Replace a primitive's placeholder accessors with decompressed draco data.

    The accessors of a draco-compressed primitive have no `bufferView`, so the
    loader filled them with zeros before calling us. All of the geometry is in
    a single opaque blob, and the extension carries the indirection we need to
    unpack it: a mapping of glTF attribute name to draco attribute id.

    Parameters
    ----------
    context
      PrimitivePreprocessContext, whose `accessors` we mutate in-place.
    """
    import DracoPy

    data = context["data"]
    accessors = context["accessors"]
    attributes = context["primitive"].get("attributes", {})

    # one blob holds every compressed attribute for this primitive
    decoded = DracoPy.decode(context["views"][data["bufferView"]])

    # the extension stores name -> draco id and we look up the other way
    names = {ident: name for name, ident in data["attributes"].items()}

    # overwrite the zero placeholders in-place by accessor index
    for attr in decoded.attributes:
        name = names.get(attr["unique_id"])
        if name in attributes:
            accessors[attributes[name]] = attr["data"]

    # indices aren't an attribute so they're not in the extension mapping
    indices = context["primitive"].get("indices")
    faces = getattr(decoded, "faces", None)
    if indices is not None and faces is not None:
        accessors[indices] = faces


@register_handler("KHR_draco_mesh_compression", scope="primitive_export")
def _draco_encode(context: PrimitiveExportContext) -> bool | None:
    """
    Compress a primitive's geometry into a single draco buffer.

    Every array in `arrays` is absorbed, so the exporter left their accessors
    with no `bufferView` rather than storing the same data twice. Returning
    `None` makes it store them after all, so a failure here exports the
    primitive uncompressed rather than pointing at accessors full of zeros.

    Parameters
    ----------
    context
      PrimitiveExportContext, whose `primitive` and `buffer_items` we mutate.

    Returns
    -------
    compressed
      True if the geometry is now inside a draco buffer, None if not.
    """
    import DracoPy

    from . import _buffer_append

    primitive = context["primitive"]
    attributes = primitive["attributes"]
    # the arrays the accessors were built from keyed by accessor index, so we
    # never unpack bytes back into numpy and can only claim what was recorded
    arrays = context["arrays"]

    # the optional attributes this primitive actually has: anything the
    # exporter stored itself, like a custom `_ATTRIBUTE`, is not in `arrays`
    absorb, optional = [], {}
    for name, keyword, kind in _draco_optional:
        value = arrays.get(attributes.get(name))
        if value is None:
            continue
        absorb.append((name, kind))
        # DracoPy asserts on float64 for UV and normals, colors stay uint8
        optional[keyword] = value if value.dtype == np.uint8 else value.astype(np.float64)

    position = arrays[attributes["POSITION"]]
    indices = arrays[primitive["indices"]]

    # `preserve_order` is load-bearing: without it draco rewelds and permutes
    # vertices, which would invalidate the count/min/max already written into
    # every accessor and silently misindex any attribute we didn't compress
    buffer = DracoPy.encode(
        points=position,
        faces=indices,
        preserve_order=True,
        quantization_bits=_DRACO_QUANTIZATION,
        compression_level=_DRACO_COMPRESSION,
        **optional,
    )

    # decode what we just encoded, which does two things we can't get otherwise:
    # it reports the attribute ids draco assigned, and it proves the round trip
    # preserved our counts before we let the exporter skip storing the source.
    # DO NOT remove this check: it is all that stands between a lossy encoder
    # and silently corrupt geometry in the exported file.
    check = DracoPy.decode(buffer)
    if len(check.points) != len(position) or len(check.faces) != len(indices):
        log.warning("draco round trip changed the vertex count, not compressing")
        return None

    # draco identifies attributes by kind, the extension identifies them by id.
    # an attribute draco silently dropped is missing from `ids`, and the
    # `KeyError` leaves this primitive uncompressed rather than pointing at
    # an accessor the exporter is about to stop storing
    ids = {attr["attribute_type"]: attr["unique_id"] for attr in check.attributes}
    absorbed = {
        name: ids[getattr(DracoPy.AttributeType, kind)]
        for name, kind in [("POSITION", "POSITION"), *absorb]
    }

    primitive.setdefault("extensions", {})["KHR_draco_mesh_compression"] = {
        # identical geometry encodes identically so this dedupes for free
        "bufferView": _buffer_append(context["buffer_items"], buffer),
        "attributes": absorbed,
    }
    return True
