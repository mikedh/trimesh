from string import Template

import numpy as np

from ... import resources, util
from ...exchange.ply import _elements_to_kwargs, _parse_header, _ply_ascii, _ply_binary
from ...typed import Dict, Optional
from .misc import edges_to_path


def export_ply(
    path,
    encoding: Optional[str] = "binary",
) -> bytes:
    """
    Export a path in the PLY format.

    Parameters
    -----------
    path: path to export
    encoding: PLY encoding: 'ascii' or 'binary_little_endian'

    Returns
    -----------
    export: bytes of result
    """
    # evaluate input args
    # allow a shortcut for binary
    if encoding == "binary":
        encoding = "binary_little_endian"
    elif encoding not in ["binary_little_endian", "ascii"]:
        raise ValueError("encoding must be binary or ascii")

    # custom numpy dtypes for exporting
    dtype_edge = [("index", "<i4", (2))]
    dtype_vertex = [("vertex", "<f4", (3))]

    # get template strings in dict
    templates = resources.get_json("templates/ply.json")
    templates["edge"] = (
        "element edge $edge_count\nproperty int vertex1\nproperty int vertex2\n"
    )
    # start collecting elements into a string for the header
    header = [templates["intro"]]

    # check if scene has geometry
    if hasattr(path, "vertices") and hasattr(path, "entities"):

        if len(path.vertices) and path.vertices.shape[-1] != 3:
            raise ValueError("only Path3D export is supported for ply")

        entities = []
        vertices = []
        for e in path.entities:
            entity_points = len(vertices)
            discretized_path = e.discrete(path.vertices).tolist()
            for pp in range(len(discretized_path) - 1):
                entities.append((entity_points + pp, entity_points+pp + 1))
            vertices.extend(discretized_path)

        # create and populate the custom dtype for vertices
        num_vertices = len(vertices)
        vertex = np.zeros(num_vertices, dtype=dtype_vertex)
        if num_vertices:
            header.append(templates["vertex"])
            vertex["vertex"] = np.asarray(vertices, dtype=np.float32)

        # put mesh edge data into custom dtype to export
        num_edges = len(entities)
        edges = np.zeros(num_edges, dtype=dtype_edge)
        if num_edges:
            header.append(templates["edge"])
            edges["index"] = np.asarray(entities, dtype=np.int32)
    else:
        num_vertices = 0
        num_edges = 0

    header_params = {"vertex_count": num_vertices, "edge_count": num_edges, "encoding": encoding}

    header.append(templates["outro"])
    export = [Template("".join(header)).substitute(header_params).encode("utf-8")]

    if encoding == "binary_little_endian":
        if hasattr(path, "vertices"):
            export.append(vertex.tobytes())
        if hasattr(path, "entities"):
            export.append(edges.tobytes())
    elif encoding == "ascii":
        if hasattr(path, "vertices"):
            export.append(
            util.structured_array_to_string(vertex, col_delim=" ", row_delim="\n").encode(
                "utf-8"
            ),
        )

        if hasattr(path, "entities"):
            export.extend(
                [
                    b"\n",
                    util.structured_array_to_string(
                        edges, col_delim=" ", row_delim="\n"
                    ).encode("utf-8"),
                ]
            )
        export.append(b"\n")
    else:
        raise ValueError("encoding must be ascii or binary!")

    return b"".join(export)


def load_ply(file_obj, **kwargs) -> Dict:
    """
    Load a PLY file to a dictionary containing vertices and entities

    Parameters
    -----------
    file_obj: file or file-like object (has object.read method)

    Returns
    -----------
    result: keys are entities, vertices and metadata
    """

    # OrderedDict which is populated from the header
    elements, is_ascii, image_name = _parse_header(file_obj)

    # functions will fill in elements from file_obj
    if is_ascii:
        _ply_ascii(elements, file_obj)
    else:
        _ply_binary(elements, file_obj)

    # translate loaded PLY elements to kwargs
    kwargs = _elements_to_kwargs(image=None, elements=elements, fix_texture=False)

    try:
        v1 = kwargs["metadata"]["_ply_raw"]["edge"]["data"]["vertex1"]
        v2 = kwargs["metadata"]["_ply_raw"]["edge"]["data"]["vertex2"]
        result = edges_to_path(np.column_stack((v1, v2)), kwargs["vertices"])
    except KeyError:
        # special case for empty files
        result = kwargs
        result.update({"entities": [], "geometry": None})
        if "vertices" not in kwargs:
            result.update({"vertices": np.asarray([], dtype=np.float32)})

    # return result as kwargs for Path constructor
    result.update({"metadata": {}})

    return result
