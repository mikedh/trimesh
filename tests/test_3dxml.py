try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np


class DXMLTest(g.unittest.TestCase):
    def test_abaqus_texture(self):
        # an assembly with instancing
        s = g.get_mesh("cube1.3dxml")

        # should be 1 unique meshes
        assert len(s.geometry) == 1

        v = s.geometry["Abaqus_Geometry"].visual
        assert v.kind == "texture"
        assert len(np.unique(v.to_color().vertex_colors, axis=0)) == 4

    def test_abaqus_blocks(self):
        # an assembly with two Faces elements of different color
        s = g.get_mesh("blocks.3dxml")
        assert g.np.isclose(s.volume, 18000, atol=1.0)
        v = s.geometry["Abaqus_Geometry"].visual
        assert v.kind == "face"
        assert len(np.unique(v.face_colors, axis=0)) == 2


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
