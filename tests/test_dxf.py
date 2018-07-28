try:
    from . import generic as g
except BaseException:
    import generic as g


class DXFTest(g.unittest.TestCase):

    def test_dxf(self):
        drawings = g.get_2D()

        # split drawings into single body parts
        splits = []
        for d in drawings:
            s = d.split()
            # check area of split result vs source
            assert g.np.isclose(sum(i.area for i in s),
                                d.area)
            splits.append(s)

            d.export(file_obj='temp.dxf')
            r = g.trimesh.load('temp.dxf')
            assert g.np.isclose(r.area, d.area)

        single = g.np.hstack(splits)

        for p in single:
            p.vertices /= p.scale
            p.export(file_obj='temp.dxf')
            r = g.trimesh.load('temp.dxf')
            ratio = abs(p.length - r.length) / p.length
            if ratio > .01:
                g.log.error('perimeter ratio on export %s wrong! %f %f %f',
                            p.metadata['file_name'],
                            p.length,
                            r.length,
                            ratio)

                raise ValueError('perimeter ratio too large ({}) on {}'.format(
                    ratio,
                    p.metadata['file_name']))

    def test_spline(self):

        d = g.get_mesh('2D/cycloidal.dxf')

        assert len(d.entities) == 1
        assert type(d.entities[0]).__name__ == 'BSpline'

        # export to dxf and wrap as a file object
        e = g.trimesh.util.wrap_as_stream(d.export(file_type='dxf'))
        # reconstitute drawing
        r = g.trimesh.load(e, file_type='dxf')

        # make sure reconstituted drawing is the same as the source
        assert len(r.entities) == 1
        assert type(r.entities[0]).__name__ == 'BSpline'
        assert g.np.isclose(r.area, d.area)

        assert len(d.entities[0].points) == len(r.entities[0].points)
        assert len(d.entities[0].knots) == len(r.entities[0].knots)

    def test_xrecord(self):
        # data to store to test export / import round trip
        data = {'thangs': 'poppin',
                'pnts': g.np.arange(9).reshape((-1, 3))}

        # get a drawing and add our data to metadata
        d = g.get_mesh('2D/wrench.dxf')
        d.metadata.update(data)

        # export as a DXF file, which should put our
        # custom data into an XRecord
        d.export('hey.dxf')

        # reload from export
        r = g.trimesh.load('hey.dxf')

        # check numpy round trip
        assert g.np.allclose(r.metadata['pnts'],
                             data['pnts'])
        # check string roundtrip
        assert r.metadata['thangs'] == 'poppin'


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
