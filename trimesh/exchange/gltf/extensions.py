"""
gltf_extensions.py
------------------

Extension registry for glTF import/export with scope-based handlers.
"""

from typing import Any, Callable

from ...constants import log
from ...typed import Dict, List, Literal, Optional

# Scopes define where in the glTF load process handlers run:
#   material            - after parsing material, can override PBR values
#   texture_source      - when resolving texture image index
#   primitive           - after loading primitive, can add face_attributes
#   primitive_preprocess - before accessor reads, can modify accessors in-place
Scope = Literal["material", "texture_source", "primitive", "primitive_preprocess"]

# Handler type aliases for each scope
Handler = Callable[..., Any]

# callback to parse material dict and resolve texture references
# signature: (*, data: Dict) -> Dict
ParseTextures = Callable[..., Dict[str, Any]]

# Registry: {scope: {extension_name: handler}}
_handlers: Dict[str, Dict[str, Handler]] = {}


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
    """
    if scope not in _handlers:
        _handlers[scope] = {}

    def decorator(func: Handler) -> Handler:
        _handlers[scope][name] = func
        return func

    return decorator


def handle_extensions(
    *,
    extensions: Optional[Dict[str, Any]],
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
      Handler scope to invoke, e.g. "material", "texture_source".
    **kwargs
      Additional arguments to pass to handlers. Required kwargs by scope:
        - material: parse_textures, images
        - texture_source: (none)
        - primitive: primitive, mesh_kwargs, accessors
        - primitive_preprocess: primitive, accessors

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
            if result := _handlers[scope][ext_name](data=data, **kwargs):
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
            # merge extension results, trusting extensions to provide appropriate data
            for key, value in ext_result.items():
                if isinstance(value, dict):
                    # merge dict values, like metadata
                    mesh_kwargs.setdefault(key, {}).update(value)
                else:
                    # overwrite non-dict values
                    mesh_kwargs[key] = value

    return results


@register_handler("KHR_materials_pbrSpecularGlossiness", scope="material")
def _specular_glossiness(
    *,
    data: Dict[str, Any],
    parse_textures: ParseTextures,
    images: List,
) -> Optional[Dict[str, Any]]:
    """
    Convert specular-glossiness material to PBR metallic-roughness.

    Parameters
    ----------
    data
      KHR_materials_pbrSpecularGlossiness extension data.
    parse_textures
      Function to parse material values and resolve texture references.
    images
      List of parsed texture images.

    Returns
    -------
    pbr_dict
      PBR metallic-roughness parameters, or None on failure.
    """
    try:
        from ...visual.gloss import specular_to_pbr

        return specular_to_pbr(**parse_textures(data=data))
    except Exception:
        log.debug("failed to convert specular-glossiness", exc_info=True)
        return None


@register_handler("EXT_texture_webp", scope="texture_source")
def _texture_webp_source(*, data: Dict[str, Any]) -> Optional[int]:
    """
    Return image source index from EXT_texture_webp.

    Parameters
    ----------
    data
      EXT_texture_webp extension data with "source" key.

    Returns
    -------
    source_index
      Index into glTF images array, or None if not present.
    """
    return data.get("source")
