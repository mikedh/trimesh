"""
gltf_extensions.py
------------------

Extension registry for glTF import/export with scope-based handlers.
Each scope has a TypedDict defining the context passed to handlers.
"""

from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np

from ...constants import log
from ...typed import NDArray

# Scopes define where in the glTF load/export process handlers run:
#   material            - after parsing material, can override PBR values
#   texture_source      - when resolving texture image index
#   primitive           - after loading primitive, can add face_attributes
#   primitive_preprocess - before accessor reads, can modify accessors in-place
#   primitive_export    - during mesh export, can compress/modify primitive data
Scope: TypeAlias = Literal[
    "material", "texture_source", "primitive", "primitive_preprocess", "primitive_export"
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
    buffer_items: dict
    primitive: dict
    # the arrays the primitive's accessors were built from, in the dtype they
    # would have been written with, keyed by accessor index: a handler storing
    # them itself never unpacks bytes back into numpy, and the exporter stores
    # them after all if no handler claims them
    arrays: dict[int, NDArray]


# Handler type alias - handlers receive a context dict
Handler: TypeAlias = Callable[[Any], Any]

# callback to parse material dict and resolve texture references
# signature: (*, data: dict) -> dict
ParseTextures: TypeAlias = Callable[..., dict[str, Any]]

# Registry: {scope: {extension_name: handler}}
_handlers: dict[str, dict[str, Handler]] = {}


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
    if scope not in _handlers:
        _handlers[scope] = {}

    def decorator(func: Handler) -> Handler:
        _handlers[scope][name] = func
        return func

    return decorator


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
) -> Any:
    """
    Process extensions dict for a given scope, calling registered handlers.

    Parameters
    ----------
    extensions
      The "extensions" dict from a glTF element, or None.
    scope
      Handler scope to invoke.
    **kwargs
      Scope-specific arguments that will be combined with extension data
      into a typed context dict. Required kwargs by scope:
        - material: parse_textures, images
        - texture_source: (none)
        - primitive: primitive, mesh_kwargs, accessors
        - primitive_preprocess: primitive, accessors, views
        - primitive_export: mesh, name, tree, buffer_items, primitive, arrays

    Returns
    -------
    results
      Dict of {extension_name: result} for most scopes.
      For scopes ending in "_source", returns first non-None result.
      For "primitive" scope, automatically merges results into mesh_kwargs.
    """
    if not extensions or scope not in _handlers:
        return {} if not scope.endswith("_source") else None

    results = {}
    for ext_name, data in extensions.items():
        if ext_name not in _handlers[scope]:
            continue
        try:
            # Build context dict with data + all kwargs
            context = {"data": data, **kwargs}
            if (result := _handlers[scope][ext_name](context)) is not None:
                results[ext_name] = result
        except Exception as e:
            log.warning(f"failed to process extension {ext_name}: {e}")

    # for _source scopes return first result, otherwise return all results
    if scope.endswith("_source"):
        return next(iter(results.values()), None)

    # for primitive scope, automatically merge results into mesh_kwargs
    if scope == "primitive" and "mesh_kwargs" in kwargs:
        mesh_kwargs = kwargs["mesh_kwargs"]
        for ext_result in results.values():
            if not isinstance(ext_result, dict):
                continue
            # merge extension results, recursively merging nested dicts
            for key, value in ext_result.items():
                if isinstance(value, dict):
                    if key not in mesh_kwargs:
                        mesh_kwargs[key] = {}
                    _deep_merge(mesh_kwargs[key], value)
                else:
                    mesh_kwargs[key] = value

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
