try:
    from . import generic as g
except BaseException:
    import generic as g


def bounds_no_overlap(bounds, epsilon=1e-8):
    """
    Check that a list of axis-aligned bounding boxes
    contains no overlaps using `rtree`.

    Parameters
    ------------
    bounds : (n, 2, dimension) float
      Axis aligned bounding boxes
    epsilon : float
      Amount to shrink AABB to avoid spurious floating
      point hits.

    Returns
    --------------
    not_overlap : bool
      True if no bound intersects any other bound.
    """
    # pad AABB by epsilon for deterministic intersections
    padded = g.np.array(bounds) + g.np.reshape([epsilon, -epsilon],
                                               (1, 2, 1))
    tree = g.trimesh.util.bounds_tree(padded)
    # every returned AABB should not overlap with any other AABB
    return all(set(tree.intersection(current.ravel())) ==
               {i} for i, current in enumerate(bounds))


def transforms_match(bounds, extents, transforms):
    """
    Check to see if transforms match.
    """
    assert len(bounds) == len(extents)
    assert len(bounds) == len(transforms)
    box = g.trimesh.creation.box

    for b, t, e in zip(bounds, transforms, extents):
        # create a box with the placed bounds
        a = box(bounds=b)
        # create a box using the roll transform
        b = box(extents=e, transform=t)

        # they should be identical
        if not g.np.allclose(a.bounds, b.bounds):
            return False
    return True


def _solid_image(color, size):
    """
    Return a PIL image that is all one color.

    Parameters
    ------------
    color : (4,) uint8
      RGBA color
    size : (2,) int
      Size of solid color image

    Returns
    -----------
    solid : PIL.Image
      Image with requested color and size.
    """
    from PIL import Image
    # convert to RGB uint8
    color = g.np.array(color, dtype=g.np.uint8)[:3]

    # create a one pixel RGB image
    image = Image.fromarray(
        g.np.tile(color, (g.np.product(size), 1)).reshape(
            (size[0], size[1], 3)))
    assert image.size == tuple(size[::-1])

    return image


class PackingTest(g.unittest.TestCase):

    def test_obb(self):
        from trimesh.path import packing
        nestable = [g.Polygon(i) for i in g.data['nestable']]
        inserted, transforms = packing.polygons(nestable)

    def test_image(self):
        from trimesh.path import packing

        images = [_solid_image([255, 0, 0, 255], [10, 10]),
                  _solid_image([0, 255, 0, 255], [120, 12]),
                  _solid_image([0, 0, 255, 255], [144, 500])]

        p, offset = packing.images(images, power_resize=False)
        # result should not be a power-of-two size
        assert not g.np.allclose(g.np.log2(p.size) % 1.0, 0.0)
        assert g.np.isfinite(offset).all()

        p, offset = packing.images(images, power_resize=True)
        assert g.np.allclose(g.np.log2(p.size) % 1.0, 0.0)
        assert g.np.isfinite(offset).all()

    def test_paths(self):
        from trimesh.path import packing
        nestable = [g.Polygon(i) for i in g.data['nestable']]
        paths = [g.trimesh.load_path(i) for i in nestable]

        with g.Profiler() as P:
            r, inserted = packing.paths(paths)
        print(P.output_text())

        # number of paths inserted
        count = len(g.np.unique(inserted))
        # should have inserted all our paths
        assert count == len(paths)
        # splitting should result in the right number of paths
        assert count == len(r.split())

    def test_3D(self):
        from trimesh.path import packing
        e = g.np.array([[14., 14., 0.125],
                        [13.84376457, 13.84376457, 0.25],
                        [14., 14., 0.125],
                        [12.00000057, 12.00000057, 0.25],
                        [14., 14., 0.125],
                        [12.83700787, 12.83700787, 0.375],
                        [12.83700787, 12.83700787, 0.125],
                        [14., 14., 0.625],
                        [1.9999977, 1.9999509, 0.25],
                        [0.87481696, 0.87463294, 0.05],
                        [0.99955503, 0.99911677, 0.1875]])

        # try packing these 3D boxes
        bounds, consume = packing.rectangles_single(e)
        assert consume.all()
        # assert all bounds are well constructed
        assert bounds_no_overlap(bounds)

        # try packing these 3D boxes
        bounds, consume = packing.rectangles_single(e, size=[14, 14, 1])
        assert not consume.all()
        # assert all bounds are well constructed
        assert bounds_no_overlap(bounds)

    def test_transform(self):
        from trimesh.path import packing
        # try in 3D with random OBB and orientation
        ori = g.np.array([[14., 14., 0.125],
                          [13.84376457, 13.84376457, 0.25],
                          [14., 14., 0.125],
                          [12.00000057, 12.00000057, 0.25],
                          [14., 14., 0.125],
                          [12.83700787, 12.83700787, 0.375],
                          [12.83700787, 12.83700787, 0.125],
                          [14., 14., 0.625],
                          [1.9999977, 1.9999509, 0.25],
                          [0.87481696, 0.87463294, 0.05],
                          [0.99955503, 0.99911677, 0.1875]])

        density = []
        with g.Profiler() as P:
            for i in range(10):
                # roll the extents by a random amount and offset
                extents = []
                for i in ori:
                    extents.append(
                        g.np.roll(i, int(g.np.random.random() * 10)) +
                        g.np.random.random(3))
                extents = g.np.array(extents)

                bounds, consume = packing.rectangles(extents)
                # should have inserted everything because we didn't specify
                # a maximum `size` to packing
                assert consume.all()
                assert len(bounds) == consume.sum()
                assert bounds_no_overlap(bounds)

                # generate the transforms for the packing
                transforms = packing.roll_transform(
                    bounds=bounds, extents=extents)

                assert transforms_match(bounds=bounds,
                                        extents=extents[consume],
                                        transforms=transforms)

                viz = packing.visualize(bounds=bounds, extents=extents)
                density.append(viz.volume / viz.bounding_box.volume)

                bounds, consume = packing.rectangles(
                    extents, size=[16, 16, 10])
                # should have inserted everything because we didn't specify
                # a maximum `size` to packing
                assert bounds_no_overlap(bounds)

                # generate the transforms for the packing
                transforms = packing.roll_transform(
                    bounds=bounds, extents=extents[consume])
                assert transforms_match(bounds=bounds,
                                        extents=extents[consume],
                                        transforms=transforms)
                viz = packing.visualize(
                    bounds=bounds, extents=extents[consume])
                density.append(viz.volume / viz.bounding_box.volume)

                bounds, consume = packing.rectangles(
                    extents, size=[16, 16, 10], rotate=False)
                # should have inserted everything because we didn't specify
                # a maximum `size` to packing
                assert bounds_no_overlap(bounds)
                # generate the transforms for the packing
                transforms = packing.roll_transform(
                    bounds=bounds, extents=extents[consume])
                assert transforms_match(bounds=bounds,
                                        extents=extents[consume],
                                        transforms=transforms)
                viz = packing.visualize(
                    bounds=bounds, extents=extents[consume])
                density.append(viz.volume / viz.bounding_box.volume)

                bounds, consume = packing.rectangles(extents, rotate=False)
                # should have inserted everything because we didn't specify
                # a maximum `size` to packing
                assert bounds_no_overlap(bounds)
                # generate the transforms for the packing
                transforms = packing.roll_transform(
                    bounds=bounds, extents=extents[consume])
                assert transforms_match(bounds=bounds,
                                        extents=extents[consume],
                                        transforms=transforms)
                viz = packing.visualize(
                    bounds=bounds, extents=extents[consume])
                density.append(viz.volume / viz.bounding_box.volume)
        print(P.output_text())


if __name__ == '__main__':
    g.trimesh.util.attach_to_log()
    g.unittest.main()
