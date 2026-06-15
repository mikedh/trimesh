"""
trimesh.exchange.common
-----------------------

Helpers shared across exchange loaders.
"""

from typing import TypedDict


class XMLParserOptions(TypedDict):
    """lxml parser options passed to `etree.XMLParser` and `etree.iterparse`."""

    resolve_entities: bool
    no_network: bool
    huge_tree: bool
    load_dtd: bool
    dtd_validation: bool
    attribute_defaults: bool
    recover: bool


# lxml parser options shared across exchange loaders — disable entity
# resolution, network access, and DTD loading, and keep libxml2 size guards
XML_PARSER_OPTIONS: XMLParserOptions = {
    "resolve_entities": False,
    "no_network": True,
    "huge_tree": False,
    "load_dtd": False,
    "dtd_validation": False,
    "attribute_defaults": False,
    "recover": False,
}
