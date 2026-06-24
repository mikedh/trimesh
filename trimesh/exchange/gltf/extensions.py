"""
gltf_extensions.py
------------------

Extension registry for glTF import/export with scope-based handlers.
Each scope has a TypedDict defining the context passed to handlers.
"""

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np

from ...caching import hash_fast
from ...constants import log
from ...exceptions import ExceptionWrapper

try:
    import DracoPy as dpy
except BaseException as E:
    dpy = exceptions.ExceptionWrapper(E)

# GL geometry modes
_GL_TRIANGLES = 4
_GL_STRIP = 5

# GLTF data formats: numpy shapes
_shapes = {
    "SCALAR": 1,
    "VEC2": (2),
    "VEC3": (3),
    "VEC4": (4),
    "MAT2": (2, 2),
    "MAT3": (3, 3),
    "MAT4": (4, 4),
}

# GLTF data type codes: little endian numpy dtypes
_dtypes = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}

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
    buffer_items: OrderedDict
    primitive: dict
    include_normals: bool


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
        - primitive_export: mesh, name, tree, buffer_items, primitive, include_normals

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
            log.exception(f"failed to process extension {ext_name}: {e}")

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

@register_handler("KHR_draco_mesh_compression", scope="primitive_preprocess")
def _draco_mesh_compression(context: PrimitivePreprocessContext) -> None:
    """
    Decompress draco mesh data.

    Parameters
    ----------
    context
        PrimitivePreprocessContext with extension data.
    """
    primitive = context["primitive"]

    if primitive.get("mode") not in [_GL_STRIP, _GL_TRIANGLES]:
        return

    extensions = primitive.get("extensions")
    if not extensions:
        return

    extension = extensions.get("KHR_draco_mesh_compression")
    if not extension:
        return

    all_attributes = primitive["attributes"]
    extension_attributes = {v: k for k, v in extension["attributes"].items()}
    dpy_mesh = dpy.decode_buffer_to_mesh(context["views"][int(extension["bufferView"])])

    # Update any accessors with decompressed data
    for attr in dpy_mesh.attributes:
        uid = attr["unique_id"]
        if uid not in extension_attributes:
            continue
        attr_name = extension_attributes[uid]
        if attr_name not in all_attributes:
            continue
        context["accessors"][all_attributes[attr_name]] = attr["data"]

    # Handle indexed accesssors
    indices = primitive.get("indices")
    if indices is None:
        return
    context["accessors"][indices] = dpy_mesh.faces


@register_handler("KHR_draco_mesh_compression", scope="primitive_export")
def _draco_mesh_compression(context: PrimitiveExportContext) -> None:
    """
    Decompress draco mesh data.

    Parameters
    ----------
    context
        PrimitiveExportContext with extension data.
    """
    primitive = context.get("primitive")
    if not primitive:
        return

    tree = context.get("tree")
    if not tree:
        return

    buffer_items = context.get("buffer_items")
    if not buffer_items:
        return

    accessors = tree.get("accessors", [])

    accessor_idx_map = {i: v for i, v in enumerate(accessors.values())}
    accessor_key_map = {i: k for i, k in enumerate(accessors)}
    buffer_idx_map = {i: v for i, v in enumerate(buffer_items.values())}
    buffer_key_map = {i: k for i, k in enumerate(buffer_items)}

    points = None
    faces = None
    colors = None
    tex_coord = None
    normals = None
    generic_attributes = {}

    buffer_indices = []

    attributes = primitive.get("attributes")
    all_attribs = attributes.copy()
    all_attribs["FACES"] = primitive.get("indices")

    for attribute_name, attribute_idx in all_attribs.items():
        accessor = accessor_idx_map[attribute_idx]
        dtype = np.dtype(_dtypes[accessor["componentType"]])
        count = accessor["count"]
        per_item = _shapes[accessor["type"]]
        shape = (count, per_item)
        per_count = np.abs(np.prod(per_item))
        buffer_idx = accessor["bufferView"]
        start = accessor.get("byteOffset", 0)
        data = buffer_idx_map[buffer_idx]
        length = dtype.itemsize * count * per_count
        npdata = np.frombuffer(data[start : start + length], dtype=dtype).reshape(shape)
        buffer_indices.append(buffer_idx)

        match attribute_name:
            case "POSITION":
                points = npdata
            case "TEXCOORD_0":
                tex_coord = npdata.astype(float)
            case "COLOR_0":
                colors = npdata
            case "NORMAL":
                normals = npdata.astype(float)
            case "FACES":
                faces = npdata.reshape(-1, 3)
            case _:
                generic_attributes[attribute_name] = npdata

    # Compress the mesh, then decompress it to get relative sizes
    buf = dpy.encode(
        points=points,
        faces=faces,
        colors=colors,
        tex_coord=tex_coord,
        normals=normals,
        generic_attributes=generic_attributes or None,
        quantization_bits=14,  # blender defaults
        compression_level=6,   # blender defaults
    )
    dpy_mesh = dpy.decode_buffer_to_mesh(buf)

    # Edit attributes in place, removing everything but size and dtype info
    new_attribute_map = {}
    for attribute_name, attribute_idx in all_attribs.items():
        accessor = accessor_idx_map[attribute_idx]
        dtype = accessor["componentType"]
        rtype = accessor["type"]
        count = accessor["count"]

        attribute_type = None
        new_attribute_idx = None
        match attribute_name:
            case "POSITION":
                count = len(dpy_mesh.points)
                attribute_type = dpy.AttributeType.POSITION
            case "TEXCOORD_0":
                count = len(dpy_mesh.tex_coord)
                attribute_type = dpy.AttributeType.TEX_COORD
            case "COLOR_0":
                count = len(dpy_mesh.colors)
                attribute_type = dpy.AttributeType.COLOR
            case "NORMAL":
                count = len(dpy_mesh.normals)
                attribute_type = dpy.AttributeType.NORMAL
            case "FACES":
                count = len(dpy_mesh.faces) * 3
            case _:
                attr = dpy_mesh.get_attribute_by_name(attribute_name)
                count = len(attr["data"])
                new_attribute_idx = attr["unique_id"]

        if new_attribute_idx is None and attribute_type is not None:
            attr = dpy_mesh.get_attribute_by_type(attribute_type)
            new_attribute_idx = attr["unique_id"]

        if new_attribute_idx is not None:
            new_attribute_map[attribute_name] = new_attribute_idx

        new_accessor = {
            "componentType": dtype,
            "type": rtype,
            "count": count,
        }
        if "min" in accessor:
            new_accessor["min"] = accessor["min"]
        if "max" in accessor:
            new_accessor["max"] = accessor["max"]

        accessors[accessor_key_map[attribute_idx]] = new_accessor

    # Remove all referenced buffers and then add a new one containing the mesh data
    for buffer_idx in buffer_indices:
        buffer_items.pop(buffer_key_map[buffer_idx])
    buffer_items[hash_fast(buf)] = buf

    # Update extension
    if "extensions" not in primitive:
        primitive["extensions"] = {}
    primitive["extensions"]["KHR_draco_mesh_compression"] = {
        "bufferView": len(buffer_items) - 1,
        "attributes": new_attribute_map,
    }
