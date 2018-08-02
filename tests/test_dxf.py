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
        d.export('hey.dxf', include_metadata=True)

        # reload from export
        r = g.trimesh.load('hey.dxf')

        # check numpy round trip
        assert g.np.allclose(r.metadata['pnts'],
                             data['pnts'])
        # check string roundtrip
        assert r.metadata['thangs'] == 'poppin'

    def test_versions(self):
        """
        DXF files have a bajillion versions, so test against
        the same files saved in multiple versions by 2D CAD packages.
        """
        # directory where multiple versions of DXF are
        dir_versions = g.os.path.join(g.dir_2D, 'versions')

        # load the different versions
        paths = {}
        for f in g.os.listdir(dir_versions):
            # full path including directory
            ff = g.os.path.join(dir_versions, f)
            try:
                paths[f] = g.trimesh.load(ff)
            except ValueError as E:
                # something like 'r14a' for ascii
                # and 'r14b' for binary
                version = f.split('.')[-2]
                # we should only get ValueErrors on binary DXF
                assert version[-1] == 'b'
                print(E, f)

        # group drawings which have the same geometry
        # but exported in different revisions of the DXF format
        groups = g.collections.defaultdict(list)
        for k in paths.keys():
            # the first string before a period is the drawing name
            groups[k.split('.')[0]].append(k)

        # loop through each group of the same drawing
        for k, group in groups.items():
            # get the total length of every entity
            L = [paths[i].length for i in group]
            L = g.np.array(L, dtype=g.np.float64)

            # make sure all versions have consistent length
            assert g.np.allclose(L, L.mean(), rtol=.01)

            # count the number of entities in the path
            # this should be the same for every version
            E = g.np.array(
                [len(paths[i].entities) for i in group],
                dtype=g.np.int64)
            assert E.ptp() == 0


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
