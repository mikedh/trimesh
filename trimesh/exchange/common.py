"""
trimesh.exchange.common
-----------------------

Helpers shared across exchange loaders.
"""

# lxml parser options shared across exchange loaders — disable entity
# resolution, network access, and DTD loading, and keep libxml2 size guards
XML_PARSER_OPTIONS = {
    "resolve_entities": False,
    "no_network": True,
    "huge_tree": False,
    "load_dtd": False,
    "dtd_validation": False,
    "attribute_defaults": False,
    "recover": False,
}
