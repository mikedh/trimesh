"""
gltf_extensions.py
------------------

Extension registry for glTF import/export with scope-based handlers.
"""

from typing import Any, Callable, Dict, List, Literal, Optional

from ...constants import log

# Scopes define where in the glTF load process handlers run:
#   material            - after parsing material, can override PBR values
#   texture_source      - when resolving texture image index
#   primitive           - after loading primitive, can add face_attributes
#   primitive_preprocess - before accessor reads, can modify accessors in-place
Scope = Literal["material", "texture_source", "primitive", "primitive_preprocess"]

# Handler signatures for each scope - handlers MUST use these exact signatures
Handler = Callable[..., Optional[Any]]

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


def handle_extensions(extensions: Optional[Dict[str, Any]], scope: Scope, **kwargs) -> Any:
    """
    Process extensions dict for a given scope, calling registered handlers.

    Parameters
    ----------
    extensions
      The "extensions" dict from a glTF element, or None.
    scope
      Handler scope to invoke, e.g. "material", "texture_source".
    **kwargs
      Additional arguments passed to each handler.

    Returns
    -------
    results
      Dict of {extension_name: result} for most scopes.
      For scopes ending in "_source", returns first non-None result.
    """
    if not extensions or scope not in _handlers:
        return {} if not scope.endswith("_source") else None

    results = {}
    for ext_name, ext_data in extensions.items():
        if ext_name in _handlers[scope]:
            try:
                result = _handlers[scope][ext_name](ext_data, **kwargs)
                if result is not None:
                    # for _source scopes, return first match immediately
                    if scope.endswith("_source"):
                        return result
                    results[ext_name] = result
            except Exception as e:
                log.warning(f"failed to process extension {ext_name}: {e}")

    return results if not scope.endswith("_source") else None


@register_handler("KHR_materials_pbrSpecularGlossiness", scope="material")
def _specular_glossiness(
    ext_data: Dict,
    parse_values_and_textures: Callable[[Dict], Dict],
    images: List,
) -> Optional[Dict]:
    """
    Convert specular-glossiness material to PBR metallic-roughness.

    Parameters
    ----------
    ext_data
      KHR_materials_pbrSpecularGlossiness extension data.
    parse_values_and_textures
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

        return specular_to_pbr(**parse_values_and_textures(ext_data))
    except Exception:
        log.debug("failed to convert specular-glossiness", exc_info=True)
        return None


@register_handler("EXT_texture_webp", scope="texture_source")
def _texture_webp_source(ext_data: Dict) -> Optional[int]:
    """
    Return image source index from EXT_texture_webp.

    Parameters
    ----------
    ext_data
      EXT_texture_webp extension data with "source" key.

    Returns
    -------
    source_index
      Index into glTF images array, or None if not present.
    """
    return ext_data.get("source")
