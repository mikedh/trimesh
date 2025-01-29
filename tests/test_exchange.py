from trimesh.path.exchange.misc import linestrings_to_path
from shapely.geometry import LineString, MultiLineString


def test_linestrings_to_path():
    line = LineString([(0, 0), (1, 1), (2, 0)])

    result = linestrings_to_path(line)

    assert len(result["entities"]) == 1
    assert len(result["vertices"]) == 3


def test_multilinestrings_to_path():
    line = MultiLineString([
        LineString([(0, 0), (1, 1), (2, 0)]),
        LineString([(1, 0), (2, 1), (2, 1)])
    ])

    result = linestrings_to_path(line)

    assert len(result["entities"]) == 2
    assert len(result["vertices"]) == 6
