import numpy as np

try:
    from . import generic as g
except BaseException:
    import generic as g


class SceneTests(g.unittest.TestCase):
    def test_scene(self):
        for file_name in ["cycloidal.ply", "sphere.ply"]:
            mesh = g.get_mesh(file_name)
            if mesh.units is None:
                mesh.units = "in"

            scene_split = g.trimesh.scene.split_scene(mesh)
            scene_split.convert_units("in")
            scene_base = g.trimesh.Scene(mesh)

            # save hash of scene before concat
            pre = [scene_split.__hash__(), scene_base.__hash__()]
            # make sure hash's give the same result twice
            assert scene_split.__hash__() == pre[0]
            assert scene_base.__hash__() == pre[1]

            # __hash__ is a long int which fails isinstance in Python 2
            assert type(scene_base.__hash__()).__name__ in ("int", "long")

            # try out scene appending
            concat = scene_split + scene_base

            # make sure concat didn't mess with original scenes
            assert scene_split.__hash__() == pre[0]
            assert scene_base.__hash__() == pre[1]

            # make sure concatenate appended things, stuff
            assert len(concat.geometry) == (
                len(scene_split.geometry) + len(scene_base.geometry)
            )

            for s in [scene_split, scene_base]:
                pre = s.__hash__()
                assert len(s.geometry) > 0
                assert s.is_valid

                flattened = s.graph.to_flattened()
                g.json.dumps(flattened)
                edgelist = s.graph.to_edgelist()
                g.json.dumps(edgelist)

                assert s.bounds.shape == (2, 3)
                assert s.centroid.shape == (3,)
                assert s.extents.shape == (3,)
                assert isinstance(s.scale, float)
                assert g.trimesh.util.is_shape(s.triangles, (-1, 3, 3))
                assert len(s.triangles) == len(s.triangles_node)

                assert s.__hash__() == pre
                assert s.__hash__() is not None

                # should be some duplicate nodes
                assert len(s.duplicate_nodes) > 0

                # should be a single scene camera
                assert isinstance(s.camera, g.trimesh.scene.cameras.Camera)
                # should be autogenerated lights
                assert len(s.lights) > 0
                # all lights should be lights
                assert all(
                    isinstance(L, g.trimesh.scene.lighting.Light) for L in s.lights
                )
                # all lights should be added to scene graph
                assert all(L.name in s.graph for L in s.lights)

                # should have put a transform in scene graph
                assert s.camera.name in s.graph

                r = s.dump()

                gltf = s.export(file_type="gltf")
                assert isinstance(gltf, dict)
                assert len(gltf) > 0
                assert len(gltf["model.gltf"]) > 0

                glb = s.export(file_type="glb")
                assert len(glb) > 0
                assert isinstance(glb, bytes)

                for export_format in ["dict", "dict64"]:
                    # try exporting the scene as a dict
                    # then make sure json can serialize it
                    e = g.json.dumps(s.export(file_type=export_format))
                    # reconstitute the dict into a scene
                    r = g.trimesh.load(g.json.loads(e))

                    # make sure the extents are similar before and after
                    assert g.np.allclose(g.np.prod(s.extents), g.np.prod(r.extents))

                # move the scene to origin
                s.rezero()
                # if our cache dump was bad this will fail
                assert g.np.allclose(s.centroid, 0, atol=1e-5)

                # make sure explode doesn't crash
                s.explode()

    def test_cam_gltf(self):
        # Test that the camera is stored and loaded successfully into a Scene from a gltf.
        cam = g.trimesh.scene.cameras.Camera(fov=[60, 90], name="cam1")
        box = g.trimesh.creation.box(extents=[1, 2, 3])
        scene = g.trimesh.Scene(
            geometry=[box],
            camera=cam,
            camera_transform=np.array(
                [[0, 1, 0, -1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
        )
        with g.TemporaryDirectory() as d:
            # exports by path allow files to be written
            path = g.os.path.join(d, "tmp.glb")
            scene.export(path)
            r = g.trimesh.load(path, force="scene")

            # ensure no added nodes
            assert set(r.graph.nodes) == {"world", "geometry_0", "cam1"}
            # ensure same camera parameters and extrinsics
            assert (r.camera_transform == scene.camera_transform).all()
            assert r.camera.name == cam.name
            assert (r.camera.fov == cam.fov).all()
            assert r.camera.z_near == cam.z_near

    def test_scaling(self):
        # Test the scaling of scenes including unit conversion.

        scene = g.get_mesh("cycloidal.3DXML")

        hash_val = scene.__hash__()
        extents = scene.bounding_box_oriented.primitive.extents.copy()

        # TODO: have OBB return sorted extents
        # and adjust the transform to be correct

        factor = 10.0
        scaled = scene.scaled(factor)

        # the oriented bounding box should scale exactly
        # with the scaling factor
        assert g.np.allclose(
            scaled.bounding_box_oriented.primitive.extents / extents, factor
        )

        # check bounding primitives
        assert scene.bounding_box.volume > 0.0
        assert scene.bounding_primitive.volume > 0.0

        # we shouldn't have modified the original scene
        assert scene.__hash__() == hash_val
        assert scaled.__hash__() != hash_val

        # 3DXML comes in as mm
        assert all(m.units == "mm" for m in scene.geometry.values())
        assert scene.units == "mm"

        converted = scene.convert_units("in")

        assert g.np.allclose(
            converted.bounding_box_oriented.primitive.extents / extents,
            1.0 / 25.4,
            atol=1e-3,
        )

        # shouldn't have changed the original extents
        assert g.np.allclose(extents, scene.bounding_box_oriented.primitive.extents)

        # original shouldn't have changed
        assert converted.units == "in"
        assert all(m.units == "in" for m in converted.geometry.values())

        assert scene.units == "mm"

        # we shouldn't have modified the original scene
        assert scene.__hash__() == hash_val
        assert converted.__hash__() != hash_val

    def test_scaling_3D(self):
        scene = g.get_mesh("cycloidal.3DXML")
        extents = scene.extents.copy()

        factor = [0.2, 1.3, 3.3]
        scaled = scene.scaled(factor)

        assert g.np.allclose(scaled.extents / extents, factor)

        factor = [3.0, 3.0, 3.0]
        scaled = scene.scaled(factor)

        assert g.np.allclose(scaled.extents / extents, factor)

    def test_mixed_units(self):
        # create two boxes in a scene
        a = g.trimesh.creation.box()
        a.units = "in"

        b = g.trimesh.creation.box()
        b.units = "m"

        # mixed units should be None
        s = g.trimesh.Scene([a, b])
        assert len(s.geometry) == 2
        assert s.units is None

        # now all units should be meters and scene should report that
        a.units = "m"
        assert s.units == "m"

    def test_scaling_3D_mixed(self):
        # same as test_scaling_3D but input scene contains 2D and 3D geometry
        scene = g.get_mesh("scenes.zip", mixed=True)
        extents = scene.extents.copy()

        factor = [0.2, 1.3, 3.3]
        scaled = scene.scaled(factor)

        assert g.np.allclose(scaled.extents / extents, factor)

        factor = [3.0, 3.0, 3.0]
        scaled = scene.scaled(factor)

        assert g.np.allclose(scaled.extents / extents, factor)

    def test_add_geometry(self):
        # list-typed geometry should create multiple nodes,
        # e.g., below code is equivalent to
        #     scene.add_geometry(geometry[0], node_name='voxels')
        #     scene.add_geometry(geometry[1], parent_node_name='voxels')
        scene = g.trimesh.Scene()
        geometry = [g.trimesh.creation.box(), g.trimesh.creation.box()]
        scene.add_geometry(geometry)
        assert len(scene.graph.nodes_geometry) == 2

    def test_add_concat(self):
        # create a scene with just a box in it
        a = g.trimesh.creation.box().scene()
        # do the same but move the box first
        b = g.trimesh.creation.box().apply_translation([2, 2, 2]).scene()
        # add the second scene to the first
        a.add_geometry(b)
        assert len(b.geometry) == 1
        assert len(a.geometry) == 2
        assert len(a.graph.nodes_geometry) == 2

    def test_delete(self):
        # check to make sure our geometry delete cleans up
        a = g.trimesh.creation.icosphere()
        b = g.trimesh.creation.icosphere().apply_translation([2, 0, 0])
        s = g.trimesh.Scene({"a": a, "b": b})

        assert len(s.geometry) == 2
        assert len(s.graph.nodes_geometry) == 2
        # make sure every node has a transform
        [s.graph[n] for n in s.graph.nodes]

        # delete a geometry
        s.delete_geometry("a")
        assert len(s.geometry) == 1
        assert len(s.graph.nodes_geometry) == 1
        # if we screwed up the delete this will crash
        [s.graph[n] for n in s.graph.nodes]

    def test_dupe(self):
        m = g.get_mesh("tube.obj", merge_norm=True, merge_tex=True)

        assert m.body_count == 1

        s = g.trimesh.scene.split_scene(m)
        assert len(s.graph.nodes) == 2
        assert len(s.graph.nodes_geometry) == 1
        assert len(s.duplicate_nodes) == 1
        assert len(s.duplicate_nodes[0]) == 1

        c = s.copy()
        assert len(c.graph.nodes) == 2
        assert len(c.graph.nodes_geometry) == 1
        assert len(c.duplicate_nodes) == 1
        assert len(c.duplicate_nodes[0]) == 1

        u = s.convert_units("in", guess=True)
        assert len(u.graph.nodes_geometry) == 1
        assert len(u.duplicate_nodes) == 1
        assert len(u.duplicate_nodes[0]) == 1

    def test_3DXML(self):
        s = g.get_mesh("rod.3DXML")
        assert len(s.geometry) == 3
        assert len(s.graph.nodes_geometry) == 29

        dupe = s.duplicate_nodes
        assert len(dupe) == 3
        assert sum(len(i) for i in dupe) == 29

        # test cache dumping and survivability of bad
        # non-existent geometry specified in node_geometry
        s.graph.update(dupe[0][0], geometry="GARBAGE")
        # make sure geometry was updated
        assert s.graph[dupe[0][0]][1] == "GARBAGE"
        # get the regenerated duplicates
        dupe = s.duplicate_nodes
        assert len(dupe) == 3
        # should have been cleanly dropped
        assert sum(len(i) for i in dupe) == 28

    def test_tri(self):
        scene = g.get_mesh("cycloidal.3DXML")

        # scene should have triangles
        assert g.trimesh.util.is_shape(scene.triangles, (-1, 3, 3))
        assert len(scene.triangles_node) == len(scene.triangles)

        # node name of inserted 2D geometry
        node = scene.add_geometry(g.get_mesh("2D/wrench.dxf"))
        # should be in the graph
        assert node in scene.graph.nodes
        # should have geometry defined
        assert node in scene.graph.nodes_geometry

        # 2D geometry has no triangles
        assert node not in scene.triangles_node
        # every geometry node except the one 2D thing
        # we inserted should be in triangles_node
        assert len(set(scene.triangles_node)) == len(scene.graph.nodes_geometry) - 1

    def test_empty(self):
        m = g.get_mesh("bunny.ply")

        # not watertight so will result in empty scene
        s = g.trimesh.scene.split_scene(m)
        assert len(s.geometry) == 0

        s = s.convert_units("inches")
        n = s.duplicate_nodes
        assert len(n) == 0

    def test_zipped(self):
        # Make sure a zip file with multiple file types
        # is returned as a single scene.

        # allow mixed 2D and 3D geometry
        m = g.get_mesh("scenes.zip", mixed=True)
        assert len(m.geometry) >= 6

        assert len(m.graph.nodes_geometry) >= 10
        assert any(isinstance(i, g.trimesh.path.Path2D) for i in m.geometry.values())
        assert any(isinstance(i, g.trimesh.Trimesh) for i in m.geometry.values())

        m = g.get_mesh("scenes.zip", mixed=False)
        assert len(m.geometry) < 6

    def test_doubling(self):
        s = g.get_mesh("cycloidal.3DXML")

        # make sure we parked our car where we thought
        assert len(s.geometry) == 13

        # concatenate a scene with itself
        r = s + s

        # new scene should have twice as much geometry
        assert len(r.geometry) == (2 * len(s.geometry))

        assert g.np.allclose(s.extents, r.extents)

        # duplicate node groups should be twice as long
        set_ori = {len(i) * 2 for i in s.duplicate_nodes}
        set_dbl = {len(i) for i in r.duplicate_nodes}

        assert set_ori == set_dbl

    def test_empty_scene(self):
        # test an empty scene
        empty = g.trimesh.Trimesh([], [])
        assert empty.bounds is None
        assert empty.extents is None
        assert g.np.isclose(empty.scale, 1.0)

        # create a sphere
        sphere = g.trimesh.creation.icosphere()

        # empty scene should have None for bounds
        scene = empty.scene()
        assert scene.bounds is None

        # add a sphere to the empty scene
        scene.add_geometry(sphere)
        # bounds should now be populated
        assert scene.bounds.shape == (2, 3)
        assert g.np.allclose(scene.bounds, sphere.bounds)

    def test_transform(self):
        # check transforming scenes
        scene = g.trimesh.creation.box()
        assert g.np.allclose(scene.bounds, [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

        scene.apply_translation([1, 0, 1])
        assert g.np.allclose(scene.bounds, [[0.5, -0.5, 0.5], [1.5, 0.5, 1.5]])

    def test_material_group(self):
        # check scene is correctly grouped by materials
        s = g.get_mesh("box.obj", group_material=True)
        assert set(s.geometry.keys()) == {"Material", "SecondMaterial"}
        assert len(s.geometry["Material"].faces) == 8
        assert len(s.geometry["SecondMaterial"].faces) == 4

        # make sure our flag does something
        s = g.get_mesh("box.obj", group_material=False)
        assert set(s.geometry.keys()) != {"Material", "SecondMaterial"}

    def test_strip(self):
        m = g.get_mesh("cycloidal.3DXML")
        assert any(g.visual.kind is not None for g in m.geometry.values())
        m.strip_visuals()
        assert all(g.visual.kind is None for g in m.geometry.values())

    def test_export_concat(self):
        # Scenes exported in mesh formats should be
        # concatenating the meshes somewhere.
        original = g.trimesh.creation.icosphere(radius=0.123312)
        original_hash = original.identifier_hash

        scene = g.trimesh.Scene()
        scene.add_geometry(original)

        with g.TemporaryDirectory() as d:
            for ext in ["stl", "ply"]:
                file_name = g.os.path.join(d, "mesh." + ext)
                scene.export(file_name)
                loaded = g.trimesh.load(file_name)
                assert g.np.isclose(loaded.volume, original.volume)
        # nothing should have changed
        assert original.identifier_hash == original_hash

    def test_exact_bounds(self):
        m = g.get_mesh("cycloidal.3DXML")
        assert isinstance(m, g.trimesh.Scene)

        dump = m.to_mesh()
        assert isinstance(dump, g.trimesh.Trimesh)

        # scene bounds should exactly match mesh bounds
        assert g.np.allclose(m.bounds, dump.bounds)

    def test_concatenate_mixed(self):
        scene = g.trimesh.Scene(
            [
                g.trimesh.creation.icosphere(),
                g.trimesh.path.creation.rectangle([[0, 0], [1, 1]]),
            ]
        )

        dump = scene.to_mesh()
        assert isinstance(dump, g.trimesh.Trimesh)

    def test_append_scenes(self):
        scene_0 = g.trimesh.Scene(base_frame="not_world")
        scene_1 = g.trimesh.Scene(base_frame="not_world")

        scene_sum = g.trimesh.scene.scene.append_scenes(
            (scene_0, scene_1), common=["not_world"], base_frame="not_world"
        )

        assert scene_sum.graph.base_frame == "not_world"

    def test_scene_concat(self):
        # check that primitives get upgraded to meshes
        a = g.trimesh.Scene(
            [g.trimesh.primitives.Sphere(center=[5, 5, 5]), g.trimesh.primitives.Box()]
        )
        c = a.to_mesh()
        assert isinstance(c, g.trimesh.Trimesh)
        assert g.np.allclose(c.bounds, a.bounds)

        c = a.dump(concatenate=False)
        assert len(c) == len(a.geometry)

        # scene 2D
        scene_2D = g.trimesh.Scene(g.get_mesh("2D/250_cycloidal.DXF").split())
        concat = scene_2D.to_geometry()
        assert isinstance(concat, g.trimesh.path.Path2D)

        dump = scene_2D.dump(concatenate=False)
        assert len(dump) == len(scene_2D.geometry)
        assert all(isinstance(i, g.trimesh.path.Path2D) for i in dump)

        # all Path3D objects
        scene_3D = g.trimesh.Scene(
            [i.to_3D() for i in g.get_mesh("2D/250_cycloidal.DXF").split()]
        )

        dump = scene_3D.dump(concatenate=False)
        assert len(dump) >= 5
        assert all(isinstance(i, g.trimesh.path.Path3D) for i in dump)

        concat = scene_3D.to_geometry()
        assert isinstance(concat, g.trimesh.path.Path3D)

        mixed = list(scene_2D.geometry.values())
        mixed.extend(scene_3D.geometry.values())
        scene_mixed = g.trimesh.Scene(mixed)

        dump = scene_mixed.dump(concatenate=False)
        assert len(dump) == len(mixed)

        concat = scene_mixed.to_geometry()
        assert isinstance(concat, g.trimesh.path.Path3D)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
