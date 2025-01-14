try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np

# test only available engines by default
engines = g.trimesh.boolean.engines_available
# test all engines if all_dep is set
if g.all_dependencies:
    engines = g.trimesh.boolean._engines.keys()

engines = set(engines)


def test_boolean():
    times = {}
    for engine in engines:
        g.log.info("Testing boolean ops with engine %s", engine)

        a = g.get_mesh("ballA.off")
        b = g.get_mesh("ballB.off")
        truth = g.data["boolean"]

        tic = g.time.time()

        # do all booleans before checks so we can time the backends
        ab = a.difference(b, engine=engine)
        ba = b.difference(a, engine=engine)
        i = a.intersection(b, engine=engine)
        u = a.union(b, engine=engine)

        times[engine] = g.time.time() - tic

        assert ab.is_volume
        assert np.isclose(ab.volume, truth["difference"])

        assert np.allclose(ab.bounds[0], a.bounds[0])

        assert ba.is_volume
        assert np.isclose(ba.volume, truth["difference"])

        assert np.allclose(ba.bounds[1], b.bounds[1])

        assert i.is_volume
        assert np.isclose(i.volume, truth["intersection"])

        assert u.is_volume
        assert np.isclose(u.volume, truth["union"])

        g.log.info("booleans succeeded with %s", engine)

    g.log.info(times)


def test_multiple():
    """
    Make sure boolean operations work on multiple meshes.
    """
    for engine in engines:
        g.log.info("Testing multiple union with engine %s", engine)

        a = g.trimesh.primitives.Sphere(center=[0, 0, 0])
        b = g.trimesh.primitives.Sphere(center=[0, 0, 0.75])
        c = g.trimesh.primitives.Sphere(center=[0, 0, 1.5])

        r = g.trimesh.boolean.union([a, b, c], engine=engine)
        assert r.is_volume
        assert r.body_count == 1
        assert np.isclose(r.volume, 8.617306056726884)

        # try a multiple-difference
        d = g.trimesh.boolean.difference([a, b, c])
        assert d.is_volume
        assert r.body_count == 1
        assert np.isclose(d.volume, 2.2322826509159985)


def test_empty():
    for engine in engines:
        g.log.info("Testing empty intersection with engine %s", engine)

        a = g.trimesh.primitives.Sphere(center=[0, 0, 0])
        b = g.trimesh.primitives.Sphere(center=[5, 0, 0])

        i = a.intersection(b, engine=engine)

        assert i.is_empty


def test_boolean_manifold():
    from trimesh.boolean import _engines, boolean_manifold

    times = {}
    exists = not isinstance(_engines["manifold"], g.trimesh.exceptions.ExceptionWrapper)

    # run this test only when manifold3d is available or `all_dep` is enabled
    if exists or g.all_dependencies:
        import manifold3d

        for operation in ["union", "intersection"]:
            if operation == "union":
                # chain of icospheres
                meshes = [
                    g.trimesh.primitives.Sphere(center=[x / 2, 0, 0], subdivisions=0)
                    for x in range(100)
                ]
            else:
                # closer icospheres for non-empty-intersection
                meshes = [
                    g.trimesh.primitives.Sphere(center=[x, x, x], subdivisions=0)
                    for x in np.linspace(0, 0.5, 101)
                ]

            # the old 'serial' manifold method
            tic = g.time.time()
            manifolds = [
                manifold3d.Manifold(
                    mesh=manifold3d.Mesh(
                        vert_properties=np.array(mesh.vertices, dtype=np.float32),
                        tri_verts=np.array(mesh.faces, dtype=np.uint32),
                    )
                )
                for mesh in meshes
            ]
            result_manifold = manifolds[0]
            for manifold in manifolds[1:]:
                if operation == "union":
                    result_manifold = result_manifold + manifold
                else:  # operation == "intersection":
                    result_manifold = result_manifold ^ manifold
            result_mesh = result_manifold.to_mesh()
            old_mesh = g.trimesh.Trimesh(
                vertices=result_mesh.vert_properties, faces=result_mesh.tri_verts
            )
            times["serial " + operation] = g.time.time() - tic

            # new 'binary' method
            tic = g.time.time()
            new_mesh = boolean_manifold(meshes, operation)
            times["binary " + operation] = g.time.time() - tic

            # assert old_mesh.is_volume == new_mesh.is_volume
            assert old_mesh.body_count == new_mesh.body_count
            assert np.isclose(old_mesh.volume, new_mesh.volume)

        g.log.info(times)


def test_multiple_difference():
    """
    Check that `a - b - c - d - e` does what we expect on both
    the base class method and the function call.
    """

    # make a bunch of spheres that overlap
    center = (
        np.array(
            [
                [np.cos(theta), np.sin(theta), 0.0]
                for theta in np.linspace(0.0, np.pi * 2, 5)
            ]
        )
        * 1.5
    )
    # first sphere is centered
    spheres = [g.trimesh.creation.icosphere()]
    spheres.extend(g.trimesh.creation.icosphere().apply_translation(c) for c in center)

    for engine in engines:
        g.log.info("Testing multiple difference with engine %s", engine)

        # compute using meshes method
        diff_base = spheres[0].difference(spheres[1:], engine=engine)
        # compute using function call (should be identical)
        diff_meth = g.trimesh.boolean.difference(spheres, engine=engine)

        # both methods should produce the same result
        assert np.isclose(diff_base.volume, diff_meth.volume)
        assert diff_base.volume < spheres[0].volume

        # should have done the diff
        assert np.allclose(diff_base.extents, [1.5, 1.5, 2.0], atol=1e-8)
