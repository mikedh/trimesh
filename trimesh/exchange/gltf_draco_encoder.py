# Copyright 2018-2021 The glTF-Blender-IO authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import *
import os
from ..constants import log

def encode_primitive_draco(scenes, buffer_items, export_settings):
    """
    Handles draco compression.
    Moves position, normal and texture coordinate attributes into a Draco encoded buffer.
    """

    # Load DLL and setup function signatures.
    dll = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libextern_draco.so"))

    dll.encoderCreate.restype = c_void_p
    dll.encoderCreate.argtypes = [c_uint32]

    dll.encoderRelease.restype = None
    dll.encoderRelease.argtypes = [c_void_p]

    dll.encoderSetCompressionLevel.restype = None
    dll.encoderSetCompressionLevel.argtypes = [c_void_p, c_uint32]

    dll.encoderSetQuantizationBits.restype = None
    dll.encoderSetQuantizationBits.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]

    dll.encoderSetIndices.restype = None
    dll.encoderSetIndices.argtypes = [c_void_p, c_size_t, c_uint32, c_void_p]

    dll.encoderSetAttribute.restype = c_uint32
    dll.encoderSetAttribute.argtypes = [c_void_p, c_char_p, c_size_t, c_char_p, c_void_p]

    dll.encoderEncode.restype = c_bool
    dll.encoderEncode.argtypes = [c_void_p, c_uint8]

    dll.encoderGetEncodedVertexCount.restype = c_uint32
    dll.encoderGetEncodedVertexCount.argtypes = [c_void_p]

    dll.encoderGetEncodedIndexCount.restype = c_uint32
    dll.encoderGetEncodedIndexCount.argtypes = [c_void_p]

    dll.encoderGetByteLength.restype = c_uint64
    dll.encoderGetByteLength.argtypes = [c_void_p]

    dll.encoderCopy.restype = None
    dll.encoderCopy.argtypes = [c_void_p, c_void_p]

    # Don't encode the same primitive multiple times.
    encoded_primitives_cache = {}

    # Compress meshes into Draco buffers.
    buffer_items_list = list(buffer_items.items())
    for scene in scenes["scenes"]:
        for node in scene["nodes"]:
            __traverse_node(scenes, scenes["nodes"][node], lambda node: __encode_node(node, dll, export_settings, encoded_primitives_cache, scenes, buffer_items, buffer_items_list))
    # Release uncompressed index and attribute buffers.
    # Since those buffers may be shared across nodes, this step must happen after all meshes have been compressed.
    for scene in scenes["scenes"]:
        for node in scene["nodes"]:
            __traverse_node(scenes, scenes["nodes"][node], lambda node: __cleanup_node(node, scenes, buffer_items, buffer_items_list))
    # scenes["extensionsUsed"] = ["KHR_draco_mesh_compression"]
    # scenes["extensionsRequired"] = ["KHR_draco_mesh_compression"]


def __cleanup_node(node, scene, buffer_items, buffer_items_list):
    if "mesh" not in node:
        return

    for primitive in scene["meshes"][node["mesh"]]["primitives"]:
        if "extensions" not in primitive or primitive["extensions"]['KHR_draco_mesh_compression'] is None:
            continue

        indices = scene["accessors"][primitive["indices"]]
        buffer_items[buffer_items_list[indices["bufferView"]][0]] = bytearray()
        del indices["bufferView"]
        attributes = primitive["attributes"]
        for attr_name in attributes:
            attr = scene["accessors"][attributes[attr_name]]
            if "bufferView" in attr:
                buffer_items[buffer_items_list[attr["bufferView"]][0]] = bytearray()
                del attr["bufferView"]


def __traverse_node(scene, node, f):
    f(node)
    if "children" in node:
        for child in node["children"]:
            __traverse_node(scene, scene["nodes"][child], f)


def __encode_node(node, dll, export_settings, encoded_primitives_cache, scene, buffer_items, buffer_items_list):
    if "mesh" in node:
        log.info('Draco encoder: Encoding mesh {}.'.format(node["name"]))
        for primitive in scene["meshes"][node["mesh"]]["primitives"]:
            __encode_primitive(primitive, dll, export_settings, encoded_primitives_cache, scene, buffer_items, buffer_items_list)


def __encode_primitive(primitive, dll, export_settings, encoded_primitives_cache, scene, buffer_items, buffer_items_list):
    attributes = primitive["attributes"]
    indices = scene["accessors"][primitive["indices"]]
    # Check if this primitive has already been encoded.
    # This usually happens when nodes are duplicated in Blender, thus their indices/attributes are shared data.
    t_primitive = tuple(primitive)
    if t_primitive in encoded_primitives_cache:
        if "extensions" not in primitive:
            primitiv["extensions"] = {}
        primitive["extensions"]['KHR_draco_mesh_compression'] = encoded_primitives_cache[t_primitive]
        return

    # Only do TRIANGLES primitives
    if primitive["mode"] not in [None, 4]:
        return

    if 'POSITION' not in attributes:
        log.warn('Draco encoder: Primitive without positions encountered. Skipping.')
        return

    positions = scene["accessors"][attributes['POSITION']]
    # Skip nodes without a position buffer, e.g. a primitive from a Blender shared instance.
    if "bufferView" not in positions:
        return
    encoder = dll.encoderCreate(positions["count"])

    draco_ids = {}
    for attr_name in attributes:
        attr = scene["accessors"][attributes[attr_name]]
        draco_id = dll.encoderSetAttribute(encoder, attr_name.encode(), attr["componentType"], attr["type"].encode(), buffer_items_list[attr["bufferView"]][1])
        draco_ids[attr_name] = draco_id

    dll.encoderSetIndices(encoder, indices["componentType"], indices["count"], buffer_items_list[indices["bufferView"]][1])

    dll.encoderSetCompressionLevel(encoder, export_settings['gltf_draco_mesh_compression_level'])
    dll.encoderSetQuantizationBits(encoder,
        export_settings['gltf_draco_position_quantization'],
        export_settings['gltf_draco_normal_quantization'],
        export_settings['gltf_draco_texcoord_quantization'],
        export_settings['gltf_draco_color_quantization'],
        export_settings['gltf_draco_generic_quantization'])

    preserve_triangle_order = "targets" in primitive and len(primitive["targets"]) > 0
    if not dll.encoderEncode(encoder, preserve_triangle_order):
        log.error('Could not encode primitive. Skipping primitive.')

    byte_length = dll.encoderGetByteLength(encoder)
    encoded_data = bytes(byte_length)
    dll.encoderCopy(encoder, encoded_data)

    if "extensions" not in primitive:
        primitive["extensions"] = {}

    extension_info = {
        'bufferView': len(buffer_items),
        'attributes': draco_ids
    }
    buffer_items[len(buffer_items)] = encoded_data
    primitive["extensions"]['KHR_draco_mesh_compression'] = extension_info
    encoded_primitives_cache[tuple(primitive)] = extension_info

    # Set to triangle list mode.
    primitive["mode"] = 4

    # Update accessors to match encoded data.
    indices["count"] = dll.encoderGetEncodedIndexCount(encoder)
    encoded_vertices = dll.encoderGetEncodedVertexCount(encoder)
    for attr_name in attributes:
        scene["accessors"][attributes[attr_name]]["count"] = encoded_vertices

    dll.encoderRelease(encoder)
