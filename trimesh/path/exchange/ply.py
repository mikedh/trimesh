import numpy as np
from string import Template
from ... import resources, util
from ...exchange.ply import _parse_header, _ply_binary, _ply_ascii, _elements_to_kwargs
from ...typed import Optional, Dict
from ..entities import Line
from .misc import edges_to_path


def export_ply(
        path: "Path3D",
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

    if path.vertices.shape[-1] != 3:
        raise ValueError("only Path3D export is supported to ply")

    # custom numpy dtypes for exporting
    dtype_edge = [("index", "<i4", (2))]
    dtype_vertex = [("vertex", "<f4", (3))]

    # get template strings in dict
    templates = resources.get_json("templates/ply.json")
    templates["edge"] = "element edge $edge_count\nproperty int vertex1\nproperty int vertex2\n"
    # start collecting elements into a string for the header
    header = [templates["intro"]]

    # check if scene has geometry
    if hasattr(path, "vertices"):
        header.append(templates["vertex"])
        num_vertices = len(path.vertices)

        # create and populate the custom dtype for vertices
        vertex = np.zeros(num_vertices, dtype=dtype_vertex)
        vertex["vertex"] = path.vertices.astype(np.float32)
    else:
        num_vertices = 0

    header_params = {"vertex_count": num_vertices, "encoding": encoding}

    if hasattr(path, "entities"):
        header.append(templates["edge"])

        entities = []
        for e in path.entities:
            if isinstance(e, Line):
                for pp in range(len(e.points) - 1):
                    entities.append((e.points[pp], e.points[pp + 1]))
            else:
                raise NotImplementedError(f"type {type(e)} not supported as entities")

        # put mesh edge data into custom dtype to export
        edges = np.zeros(len(entities), dtype=dtype_edge)
        edges["index"] = np.asarray(entities, dtype=np.int32)
        header_params["edge_count"] = len(entities)

    header.append(templates["outro"])
    export = [Template("".join(header)).substitute(header_params).encode("utf-8")]

    if encoding == "binary_little_endian":
        if hasattr(path, "vertices"):
            export.append(vertex.tobytes())
        if hasattr(path, "entities"):
            export.append(edges.tobytes())
    elif encoding == "ascii":
        export.append(
            util.structured_array_to_string(vertex, col_delim=" ", row_delim="\n").encode("utf-8"), )

        if hasattr(path, "entities"):
            export.extend([
                b"\n",
                util.structured_array_to_string(edges, col_delim=" ", row_delim="\n").encode("utf-8"),
            ])
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
        result = edges_to_path([v1, v2], kwargs["vertices"])
    except KeyError:
        raise ValueError("Could not load entities!")

    # return result as kwargs for Path2D constructor
    result.update({"metadata": {}})

    return result
