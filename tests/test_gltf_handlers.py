"""
test_gltf_handlers.py
---------------------

Pin the glTF extension-handler interface that upstream packages register against.
"""

import base64

try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.exchange.gltf import extensions as ext


def minimal_gltf():
    # one-triangle glTF whose POSITION accessor is bufferless -- a placeholder
    # an extension is expected to fill -- and whose primitive carries a dummy extension
    indices = g.np.array([0, 1, 2], dtype=g.np.uint32).tobytes()
    uri = "data:application/octet-stream;base64," + base64.b64encode(indices).decode()
    tree = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                        "extensions": {"TEST_dummy": {}},
                    }
                ]
            }
        ],
        "accessors": [
            # bufferless -> created as placeholder zeros
            {"componentType": 5126, "count": 3, "type": "VEC3"},
            {"componentType": 5125, "count": 3, "type": "SCALAR", "bufferView": 0},
        ],
        "bufferViews": [{"buffer": 0, "byteLength": 12, "byteOffset": 0}],
        "buffers": [{"byteLength": 12, "uri": uri}],
    }
    return g.io.BytesIO(g.json.dumps(tree).encode())


class GLTFHandlerTest(g.unittest.TestCase):
    def setUp(self):
        # snapshot the global handler registry so tests don't leak
        self._snapshot = {scope: dict(h) for scope, h in ext._handlers.items()}

    def tearDown(self):
        ext._handlers.clear()
        ext._handlers.update(self._snapshot)

    def test_register_roundtrip(self):
        @ext.register_handler("TEST_x", scope="primitive_preprocess")
        def handler(context):
            return None

        # the decorator registers into the scope registry and returns the function
        assert ext._handlers["primitive_preprocess"]["TEST_x"] is handler

    def test_preprocess_context_keys(self):
        # the loader must pass the documented context keys to preprocess handlers
        seen = {}

        @ext.register_handler("TEST_dummy", scope="primitive_preprocess")
        def spy(context):
            seen.update({k: type(v) for k, v in context.items()})
            return None

        g.trimesh.load(minimal_gltf(), file_type="gltf")
        assert {"data", "primitive", "accessors", "views"}.issubset(seen)

    def test_no_handler_warns_and_zeros(self):
        # without a handler the placeholder geometry stays zero and we warn by name
        with self.assertLogs("trimesh", level="WARNING") as cm:
            scene = g.trimesh.load(minimal_gltf(), file_type="gltf")
        geom = next(iter(scene.geometry.values()))
        assert g.np.allclose(geom.vertices, 0.0)
        assert any("TEST_dummy" in line for line in cm.output)

    def test_handler_fills_placeholder(self):
        # a handler may fill the placeholder accessor in place -- a successful
        # decode must not warn
        filled = g.np.arange(9, dtype=g.np.float32).reshape((3, 3))

        @ext.register_handler("TEST_dummy", scope="primitive_preprocess")
        def fill(context):
            context["accessors"][context["primitive"]["attributes"]["POSITION"]][:] = (
                filled
            )
            return None

        with self.assertNoLogs("trimesh", level="WARNING"):
            scene = g.trimesh.load(minimal_gltf(), file_type="gltf")
        geom = next(iter(scene.geometry.values()))
        assert g.np.allclose(geom.vertices, filled)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
