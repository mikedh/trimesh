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
import numpy as np

def decode_primitive_draco(gltf, views, prim, access):
    """
    Handles draco compression.
    Moves decoded data into new buffers and buffer views held by the accessors of the given primitive.
    """

    # Load DLL and setup function signatures.
    dll = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libextern_draco.so"))

    dll.decoderCreate.restype = c_void_p
    dll.decoderCreate.argtypes = []

    dll.decoderRelease.restype = None
    dll.decoderRelease.argtypes = [c_void_p]

    dll.decoderDecode.restype = c_bool
    dll.decoderDecode.argtypes = [c_void_p, c_void_p, c_size_t]

    dll.decoderReadAttribute.restype = c_bool
    dll.decoderReadAttribute.argtypes = [c_void_p, c_uint32, c_size_t, c_char_p]

    dll.decoderGetVertexCount.restype = c_uint32
    dll.decoderGetVertexCount.argtypes = [c_void_p]

    dll.decoderGetIndexCount.restype = c_uint32
    dll.decoderGetIndexCount.argtypes = [c_void_p]

    dll.decoderAttributeIsNormalized.restype = c_bool
    dll.decoderAttributeIsNormalized.argtypes = [c_void_p, c_uint32]

    dll.decoderGetAttributeByteLength.restype = c_size_t
    dll.decoderGetAttributeByteLength.argtypes = [c_void_p, c_uint32]

    dll.decoderCopyAttribute.restype = None
    dll.decoderCopyAttribute.argtypes = [c_void_p, c_uint32, c_void_p]

    dll.decoderReadIndices.restype = c_bool
    dll.decoderReadIndices.argtypes = [c_void_p, c_size_t]

    dll.decoderGetIndicesByteLength.restype = c_size_t
    dll.decoderGetIndicesByteLength.argtypes = [c_void_p]

    dll.decoderCopyIndices.restype = None
    dll.decoderCopyIndices.argtypes = [c_void_p, c_void_p]

    decoder = dll.decoderCreate()
    extension = prim['extensions']['KHR_draco_mesh_compression']

    name = prim['name'] if 'name' in prim else '[unnamed]'

    # Create Draco decoder.
    draco_buffer = views[extension['bufferView']]
    if not dll.decoderDecode(decoder, draco_buffer, len(draco_buffer)):
        log.error('Draco Decoder: Unable to decode. Skipping primitive {}.'.format(name))
        return

    # Read indices.
    index_accessor = gltf['accessors'][prim['indices']]
    if dll.decoderGetIndexCount(decoder) != index_accessor['count']:
        log.warn('Draco Decoder: Index count of accessor and decoded index count does not match. Updating accessor.')
        gltf['accessors'][prim['indices']]['count'] = dll.decoderGetIndexCount(decoder)
    if not dll.decoderReadIndices(decoder, index_accessor['componentType']):
        log.error('Draco Decoder: Unable to decode indices. Skipping primitive {}.'.format(name))
        return

    indices_byte_length = dll.decoderGetIndicesByteLength(decoder)
    decoded_data = bytes(indices_byte_length)
    dll.decoderCopyIndices(decoder, decoded_data)

    cur_access = access[prim['indices']]
    access[prim['indices']] = np.frombuffer(decoded_data, dtype=cur_access.dtype).reshape(cur_access.shape)

    # Read each attribute.
    for attr_idx, attr in enumerate(extension['attributes']):
        dracoId = extension['attributes'][attr]
        if attr not in prim['attributes']:
            log.error('Draco Decoder: Draco attribute {} not in primitive attributes. Skipping primitive {}.'.format(attr, name))
            return

        accessor = gltf['accessors'][prim['attributes'][attr]]
        if dll.decoderGetVertexCount(decoder) != accessor['count']:
            log.warn('Draco Decoder: Vertex count of accessor and decoded vertex count does not match for attribute {}. Updating accessor.'.format(attr, name))
            accessor['count'] = dll.decoderGetVertexCount(decoder)
        if not dll.decoderReadAttribute(decoder, dracoId, accessor['componentType'], accessor['type'].encode()):
            log.error('Draco Decoder: Could not decode attribute {}. Skipping primitive {}.'.format(attr, name))
            return

        byte_length = dll.decoderGetAttributeByteLength(decoder, dracoId)
        decoded_data = bytes(byte_length)
        dll.decoderCopyAttribute(decoder, dracoId, decoded_data)
        cur_access = access[prim['attributes'][attr]]
        access[prim['attributes'][attr]] = np.frombuffer(decoded_data, dtype=cur_access.dtype).reshape(cur_access.shape)

    dll.decoderRelease(decoder)
    return access