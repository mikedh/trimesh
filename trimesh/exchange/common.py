"""
trimesh.exchange.common
-----------------------

Helpers shared across exchange loaders.
"""

# lxml parser options shared across exchange loaders — disable entity
# resolution, network access, and DTD loading. `huge_tree` is passed per-call
# by loaders rather than set here as it is a caller opt-in
XML_PARSER_OPTIONS = {
    "resolve_entities": False,
    "no_network": True,
    "load_dtd": False,
    "dtd_validation": False,
    "attribute_defaults": False,
    "recover": False,
}
