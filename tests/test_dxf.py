try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import ezdxf
except BaseException:
    ezdxf = None


class DXFTest(g.unittest.TestCase):
    def test_dxf(self):
        # get a path we can write
        temp_name = g.tempfile.NamedTemporaryFile(suffix=".dxf", delete=False).name
        loaded = g.get_2D()
        # split drawings into single body parts
        splits = []
        for d in loaded:
            s = d.split()
            # check area of split result vs source
            assert g.np.isclose(sum(i.area for i in s), d.area)
            splits.append(s)

            # export the drawing to the file
            d.export(file_obj=temp_name)

            # try using ezdxf as a simple validator
            # it raises exceptions aggressively
            if ezdxf is not None:
                with open(temp_name) as f:
                    ezdxf.read(f)

            # export to a string
            text = d.export(file_type="dxf")

            # DXF files are always pairs of lines
            lines = str.splitlines(str(text))
            assert (len(lines) % 2) == 0
            assert all(len(L.strip()) > 0 for L in lines)

            # reload the file by name and by stream
            rc = [
                g.trimesh.load(temp_name),
                g.trimesh.load(g.io_wrap(text), file_type="dxf"),
            ]

            # compare reloaded with original
            for r in rc:
                assert g.np.isclose(r.area, d.area)
                assert g.np.isclose(r.length, d.length, rtol=1e-4)
                assert len(r.entities) == len(d.entities)

        single = g.np.hstack(splits)
        for p in single:
            p.vertices /= p.scale

            # make sure exporting by name works
            # use tempfile to avoid dumping file in
            # our working directory
            p.export(temp_name)
            r = g.trimesh.load(temp_name)

            ratio = abs(p.length - r.length) / p.length
            if ratio > 0.01:
                g.log.error(
                    "perimeter ratio on export %s wrong! %f %f %f",
                    p.metadata["file_name"],
                    p.length,
                    r.length,
                    ratio,
                )

                raise ValueError(
                    "perimeter ratio too large ({}) on {}".format(
                        ratio, p.metadata["file_name"]
                    )
                )

    def test_spline(self):
        d = g.get_mesh("2D/cycloidal.dxf")

        assert len(d.entities) == 1
        assert type(d.entities[0]).__name__ == "BSpline"

        # export to dxf and wrap as a file object
        e = g.trimesh.util.wrap_as_stream(d.export(file_type="dxf"))
        # reconstitute drawing
        r = g.trimesh.load(e, file_type="dxf")

        # make sure reconstituted drawing is the same as the source
        assert len(r.entities) == 1
        assert type(r.entities[0]).__name__ == "BSpline"
        assert g.np.isclose(r.area, d.area)

        assert len(d.entities[0].points) == len(r.entities[0].points)
        assert len(d.entities[0].knots) == len(r.entities[0].knots)

    def test_versions(self):
        """
        DXF files have a bajillion versions, so test against
        the same files saved in multiple versions by 2D CAD
        packages.

        Version test files are named things like:
        ae.r14a.dxf: all entity types, R14 ASCII DXF
        uc.2007b.dxf: unit square, R2007 binary DXF
        """
        # directory where multiple versions of DXF are
        dir_versions = g.os.path.join(g.dir_2D, "versions")

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
                version = f.split(".")[-2]
                # we should only get ValueErrors on binary DXF
                assert version[-1] == "b"
                g.log.debug(E, f)

        # group drawings which have the same geometry
        # but exported in different revisions of the DXF format
        groups = g.collections.defaultdict(list)
        for k in paths.keys():
            # the first string before a period is the drawing name
            groups[k.split(".")[0]].append(k)

        # loop through each group of the same drawing
        for group in groups.values():
            # get the total length of every entity
            L = [paths[i].length for i in group]
            L = g.np.array(L, dtype=g.np.float64)

            # make sure all versions have consistent length
            assert g.np.allclose(L, L.mean(), rtol=0.01)

            # count the number of entities in the path
            # this should be the same for every version
            E = g.np.array([len(paths[i].entities) for i in group], dtype=g.np.int64)
            assert E.ptp() == 0

    def test_bulge(self):
        """
        Test bulged polylines which are polylines with
        implicit arcs.
        """
        # get a drawing with bulged polylines
        p = g.get_mesh("2D/LM2.dxf")
        # count the number of unclosed arc entities
        # this drawing only has polylines with bulge
        spans = [
            e.center(p.vertices)["span"]
            for e in p.entities
            if type(e).__name__ == "Arc" and not e.closed
        ]
        # should have only one outer loop
        assert len(p.root) == 1
        # should have 6 partial arcs from bulge
        assert len(spans) == 6
        # all arcs should be 180 degree slot end caps
        assert g.np.allclose(spans, g.np.pi)

    def test_text(self):
        # load file with a single text entity
        original = g.get_mesh("2D/text.dxf")

        # export then reload
        roundtrip = g.trimesh.load(
            file_obj=g.io_wrap(original.export(file_type="dxf")), file_type="dxf"
        )

        for d in [original, roundtrip]:
            # should contain a single Text entity
            assert len(d.entities) == 1

            # shouldn't crash anything
            assert len(d.polygons_closed) == 0
            assert len(d.polygons_full) == 0
            assert len(d.discrete) == 0
            assert len(d.paths) == 0

            # make sure it preserved case and special chars
            assert d.entities[0].text == "HEY WHAT's poppin"

            # height should 1.0
            assert g.np.isclose(d.entities[0].height, 1.0)

            # get the 2D rotation of the text
            angle = d.entities[0].angle(d.vertices)

            # angle should be 30 degrees
            assert g.np.isclose(angle, g.np.radians(30.0))

    def test_unicode(self):
        """
        Check our handling of unicode. Current approach is to
        just force everything into ASCII rather than handling
        the encoding flags in DXF headers.
        """
        # get a base 2D model
        m = g.get_mesh("2D/wrench.dxf")
        # make one of the entity layers a unicode string
        # store it as B64 so python2 doesn't get mad
        layer = g.base64.b64decode("VFJBw4dBRE9IT1JJWk9OVEFMX1RSQcOHQURPNA==").decode(
            "utf-8"
        )
        m.entities[0].layer = layer
        # export to a string
        export = m.export(file_type="dxf")
        # if any unicode survived the export this will fail
        export.encode("ascii")

    def test_insert_block(self):
        a = g.get_mesh("2D/insert.dxf")
        b = g.get_mesh("2D/insert_r14.dxf")

        assert len(a.polygons_full) == 2
        assert len(b.polygons_full) == 2

        assert g.np.isclose(a.area, 54075.0, atol=1)
        assert g.np.isclose(b.area, 54075.0, atol=1)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
