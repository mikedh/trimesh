import io

import numpy as np
import pytest

from trimesh.exchange.xyz import load_xyz


def test_an_eighth_column_is_not_a_color():
    loaded = load_xyz(io.StringIO("0 1 2 10 20 30 40 99\n"))
    assert loaded["colors"].shape == (1, 4)
    assert loaded["colors"].tolist() == [[10, 20, 30, 40]]


def test_seven_columns_stay_rgba():
    loaded = load_xyz(io.StringIO("0 1 2 10 20 30 40\n"))
    assert loaded["colors"].tolist() == [[10, 20, 30, 40]]


def test_six_columns_still_add_opaque_alpha():
    loaded = load_xyz(io.StringIO("0 1 2 10 20 30\n"))
    assert loaded["colors"].tolist() == [[10, 20, 30, 255]]


@pytest.mark.parametrize(
    "separator,delimiter", [(" ", None), (",", None), (",", ","), (";", ";")]
)
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("columns", [3, 6, 7])
def test_delimited_xyz(separator, delimiter, rows, columns):
    values = np.arange(rows * columns).reshape((rows, columns))
    raw = "\n".join(separator.join(map(str, row)) for row in values)
    loaded = load_xyz(io.StringIO(raw), delimiter=delimiter)
    np.testing.assert_array_equal(loaded["vertices"], values[:, :3])
    if columns == 3:
        assert loaded["colors"] is None
    else:
        expected = values[:, 3:]
        if columns == 6:
            expected = np.column_stack((expected, np.full(rows, 255)))
        np.testing.assert_array_equal(loaded["colors"], expected)
