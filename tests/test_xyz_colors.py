import io

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
