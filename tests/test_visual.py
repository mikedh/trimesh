try:
    from . import generic as g
except BaseException:
    import generic as g


class VisualTest(g.unittest.TestCase):

    def test_visual(self):
        mesh = g.get_mesh('featuretype.STL')

        self.assertFalse(mesh.visual.defined)

        for facet in mesh.facets:
            mesh.visual.face_colors[facet] = g.trimesh.visual.random_color()

        self.assertTrue(mesh.visual.defined)
        self.assertFalse(mesh.visual.transparency)

        mesh.visual.face_colors[0] = [10, 10, 10, 130]
        self.assertTrue(mesh.visual.transparency)

    def test_concatenate(self):
        a = g.get_mesh('ballA.off')
        b = g.get_mesh('ballB.off')

        a.visual.face_colors = [255, 0, 0]
        r = a + b
        self.assertTrue(any(r.visual.face_colors.ptp(axis=0) > 1))

    def test_data_model(self):
        """
        Test the probably too- magical color caching and storage system.
        """
        m = g.get_mesh('featuretype.STL')
        test_color = [255, 0, 0, 255]
        test_color_2 = [0, 255, 0, 255]
        test_color_transparent = [25, 33, 0, 146]

        # there should be nothing in the cache or DataStore when starting
        assert len(m.visual._cache) == 0
        assert len(m.visual._data) == 0
        # no visuals have been defined so this should be None
        assert m.visual.kind is None
        assert not m.visual.defined

        # this should cause colors to be generated into cache
        initial_id = id(m.visual.face_colors)
        assert m.visual.face_colors.shape[0] == len(m.faces)
        assert id(m.visual.face_colors) == initial_id
        # the values should be in the cache and not in data
        assert len(m.visual._cache) > 0
        assert len(m.visual._data) == 0
        assert not m.visual.defined
        assert not m.visual.transparency

        # this should move the color from cache to data
        m.visual.face_colors[0] = test_color
        # the operation should have moved the colors into data but
        # the object ID should be the same as on creation
        #assert id(m.visual.face_colors) == initial_id
        # the color assignment inside the array should have worked
        assert (m.visual.face_colors[0] == test_color).all()
        # the rest of the colors should be unchanged
        assert (m.visual.face_colors[1] != test_color).any()
        assert len(m.visual._data) >= 1
        assert m.visual.kind == 'face'
        assert m.visual.defined
        assert not m.visual.transparency

        # set all face colors to test color
        m.visual.face_colors = test_color
        assert (m.visual.face_colors == test_color).all()
        #assert len(m.visual._cache) == 0
        # should be just material and face information
        assert len(m.visual._data.data) >= 1
        assert m.visual.kind == 'face'
        assert bool((m.visual.vertex_colors == test_color).all())
        assert m.visual.defined
        assert not m.visual.transparency

        # this should move the color from cache to data
        m.visual.vertex_colors[0] = test_color_2
        assert (m.visual.vertex_colors[0] == test_color_2).all()
        assert (m.visual.vertex_colors[1] != test_color_2).any()
        assert m.visual.kind == 'vertex'
        assert m.visual.defined
        assert not m.visual.transparency

        m.visual.vertex_colors[1] = test_color_transparent
        assert m.visual.transparency

        test = (g.np.random.random((len(m.faces), 4)) * 255).astype(g.np.uint8)
        m.visual.face_colors = test
        assert bool((m.visual.face_colors == test).all())
        assert m.visual.kind == 'face'

        test = (g.np.random.random((len(m.vertices), 4))
                * 255).astype(g.np.uint8)
        m.visual.vertex_colors = test
        assert bool((m.visual.vertex_colors == test).all())
        assert m.visual.kind == 'vertex'

        test = (g.np.random.random(4) * 255).astype(g.np.uint8)
        m.visual.face_colors = test
        assert bool((m.visual.vertex_colors == test).all())
        assert m.visual.kind == 'face'
        m.visual.vertex_colors[0] = (
            g.np.random.random(4) * 255).astype(g.np.uint8)
        assert m.visual.kind == 'vertex'

        test = (g.np.random.random(4) * 255).astype(g.np.uint8)
        m.visual.vertex_colors = test
        assert bool((m.visual.face_colors == test).all())
        assert m.visual.kind == 'vertex'
        m.visual.face_colors[0] = (
            g.np.random.random(4) * 255).astype(g.np.uint8)
        assert m.visual.kind == 'face'

    def test_smooth(self):
        """
        Make sure cached smooth model is dumped if colors are changed
        """
        m = g.get_mesh('featuretype.STL')

        # will put smoothed mesh into visuals cache
        s = m.smoothed()
        # every color should be default color
        assert s.visual.face_colors.ptp(axis=0).max() == 0
        # set some faces to a different color
        faces = m.facets[m.facets_area.argmax()]
        m.visual.face_colors[faces] = [255, 0, 0, 255]
        # cache should be dumped yo
        s1 = m.smoothed()
        assert s1.visual.face_colors.ptp(axis=0).max() != 0

        # do the same check on vertex color
        m = g.get_mesh('featuretype.STL')
        s = m.smoothed()
        # every color should be default color
        assert s.visual.vertex_colors.ptp(axis=0).max() == 0
        m.visual.vertex_colors[g.np.arange(10)] = [255, 0, 0, 255]
        s1 = m.smoothed()
        assert s1.visual.face_colors.ptp(axis=0).max() != 0

    def test_vertex(self):

        m = g.get_mesh('torus.STL')

        m.visual.vertex_colors = [100, 100, 100, 255]

        assert len(m.visual.vertex_colors) == len(m.vertices)

    def test_conversion(self):
        m = g.get_mesh('machinist.XAML')
        assert m.visual.kind == 'face'

        # unmerge vertices so we don't get average colors
        m.unmerge_vertices()

        # store initial face colors
        initial = g.deepcopy(m.visual.face_colors.copy())

        # assign averaged vertex colors as default
        m.visual.vertex_colors = m.visual.vertex_colors
        assert m.visual.kind == 'vertex'

        m.visual._cache.clear()
        assert g.np.allclose(initial, m.visual.face_colors)

if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
