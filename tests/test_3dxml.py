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

    def test_textured_face_keeps_plain_colors(self):
        # a SolidWorks part: one color on the whole cube, an image on one
        # face whose Rep comes last; the image must not take over the part
        # and drop every other face's color
        s = g.get_mesh("solidworks_textured_face.3dxml")
        assert len(s.geometry) == 1
        m = next(iter(s.geometry.values()))
        # the color and the image are packed into one atlas
        assert m.visual.kind == "texture"
        face_colors = m.visual.to_color().vertex_colors[m.faces][:, :, :3]
        plain = (face_colors == [133, 91, 61]).all(axis=(1, 2))
        # the two triangles of the textured face sample the image
        assert plain.sum() == len(m.faces) - 2
        assert not (face_colors[~plain] == [133, 91, 61]).all(axis=2).any()

    def test_textured_face_uv_range(self):
        import io
        import zipfile

        def load(uv):
            # the same part with the UVs of its textured face replaced
            with open(
                g.os.path.join(g.dir_models, "solidworks_textured_face.3dxml"), "rb"
            ) as f:
                source = zipfile.ZipFile(f)
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as target:
                    for name in source.namelist():
                        data = source.read(name)
                        if name == "TessPart_2.3DRep":
                            data = data.replace(b">1 1,0 1,1 0,0 0<", uv)
                        target.writestr(name, data)
            buffer.seek(0)
            s = g.trimesh.load(buffer, file_type="3dxml")
            return next(iter(s.geometry.values())).visual.kind

        # an edge a hair outside 0.0-1.0 is clipped into the atlas
        assert load(b">1 1,-0.003 1,1 0,-0.003 0<") == "texture"
        # a tiled texture cannot share an atlas: keep the face colors
        assert load(b">2 2,0 2,2 0,0 0<") == "face"

    def test_face_to_vertex_colors(self):
        from trimesh.exchange.threedxml import _face_to_vertex_colors

        # two triangles of different color sharing an edge
        faces = np.array([[0, 1, 2], [2, 1, 3]])
        colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
        index, new_faces, vertex_colors = _face_to_vertex_colors(faces, colors)
        # the two shared vertices are split, one copy per color
        assert len(index) == 6
        assert (index[new_faces] == faces).all()
        assert (vertex_colors[new_faces] == colors[:, None]).all()


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
