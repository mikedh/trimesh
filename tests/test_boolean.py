try:
    from . import generic as g
except BaseException:
    import generic as g

try:
    import manifold3d
except BaseException:
    manifold3d = None

engines = [
    ("blender", g.trimesh.interfaces.blender.exists),
    ("manifold", manifold3d is not None),
]


class BooleanTest(g.unittest.TestCase):
    def setUp(self):
        self.a = g.get_mesh("ballA.off")
        self.b = g.get_mesh("ballB.off")
        self.truth = g.data["boolean"]

    def is_zero(self, value):
        return abs(value) < 0.001

    def test_boolean(self):
        a, b = self.a, self.b

        times = {}
        for engine, exists in engines:
            # if we have all_dep set it means we should fail if
            # engine is not installed so don't continue
            if not exists:
                g.log.warning("skipping boolean engine %s", engine)
                continue

            g.log.info("Testing boolean ops with engine %s", engine)

            tic = g.time.time()

            # do all booleans before checks so we can time the backends
            ab = a.difference(b, engine=engine)
            ba = b.difference(a, engine=engine)
            i = a.intersection(b, engine=engine)
            u = a.union(b, engine=engine)

            times[engine] = g.time.time() - tic

            assert ab.is_volume
            assert self.is_zero(ab.volume - self.truth["difference"])

            assert g.np.allclose(ab.bounds[0], a.bounds[0])

            assert ba.is_volume
            assert self.is_zero(ba.volume - self.truth["difference"])

            assert g.np.allclose(ba.bounds[1], b.bounds[1])

            assert i.is_volume
            assert self.is_zero(i.volume - self.truth["intersection"])

            assert u.is_volume
            assert self.is_zero(u.volume - self.truth["union"])

            g.log.info("booleans succeeded with %s", engine)

        g.log.info(times)

    def test_multiple(self):
        """
        Make sure boolean operations work on multiple meshes.
        """
        for engine, exists in engines:
            if not exists:
                continue
            a = g.trimesh.primitives.Sphere(center=[0, 0, 0])
            b = g.trimesh.primitives.Sphere(center=[0, 0, 0.75])
            c = g.trimesh.primitives.Sphere(center=[0, 0, 1.5])

            r = g.trimesh.boolean.union([a, b, c], engine=engine)

            assert r.is_volume
            assert r.body_count == 1
            assert g.np.isclose(r.volume, 8.617306056726884)

    def test_empty(self):
        for engine, exists in engines:
            if not exists:
                continue

            a = g.trimesh.primitives.Sphere(center=[0, 0, 0])
            b = g.trimesh.primitives.Sphere(center=[5, 0, 0])

            i = a.intersection(b, engine=engine)

            assert i.is_empty

    def test_boolean_manifold(self):
        if manifold3d is None:
            return

        times = {}
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
                    for x in g.np.linspace(0, 0.5, 101)
                ]

            # the old 'serial' manifold method
            tic = g.time.time()
            manifolds = [
                manifold3d.Manifold(
                    mesh=manifold3d.Mesh(
                        vert_properties=g.np.array(mesh.vertices, dtype=g.np.float32),
                        tri_verts=g.np.array(mesh.faces, dtype=g.np.uint32),
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
            new_mesh = g.trimesh.boolean.boolean_manifold(meshes, operation)
            times["binary " + operation] = g.time.time() - tic

            assert old_mesh.is_volume == new_mesh.is_volume
            assert old_mesh.body_count == new_mesh.body_count
            assert g.np.isclose(old_mesh.volume, new_mesh.volume)

        g.log.info(times)

    def test_reduce_cascade(self):
        # the multiply will explode quickly past the integer maximum

        from functools import reduce

        from trimesh.boolean import reduce_cascade

        def both(operation, items):
            """
            Run our cascaded reduce and regular reduce.
            """

            b = reduce_cascade(operation, items)

            if len(items) > 0:
                assert b == reduce(operation, items)

            return b

        for i in range(20):
            data = g.np.arange(i)
            c = both(items=data, operation=lambda a, b: a + b)

            if i == 0:
                assert c is None
            else:
                assert c == g.np.arange(i).sum()

            # try a multiply
            data = g.np.arange(i)
            c = both(items=data, operation=lambda a, b: a * b)

            if i == 0:
                assert c is None
            else:
                assert c == g.np.prod(data)

            # try a multiply
            data = g.np.arange(i)[1:]
            c = both(items=data, operation=lambda a, b: a * b)
            if i <= 1:
                assert c is None
            else:
                assert c == g.np.prod(data)

        data = ["a", "b", "c", "d", "e", "f", "g"]
        print("# reduce_pairwise\n-----------")
        r = both(operation=lambda a, b: a + b, items=data)
        assert r == "abcdefg"


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
