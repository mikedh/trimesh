import numpy as np

from ..util          import is_sequence
from .transform_tree import TransformTree

class Scene:
    '''
    A simple scene graph which can be rendered directly via pyglet/openGL,
    or through other endpoints such as a raytracer. 

    Meshes and lights are added by name, which can then be moved by updating
    transform in the transform tree. 
    '''

    def __init__(self, 
                 node      = None, 
                 base_frame ='world'):

        # instance name : mesh name
        self.instances = {}

        # dictionary, mesh name : Trimesh object
        self.meshes     = {}
        self.lights     = {}
        self.camera     = None
        self.transforms = TransformTree(base_frame = base_frame)

        self.add_mesh(node)

    def add_mesh(self, mesh):
        if mesh is None: 
            return
        elif is_sequence(mesh):
            return [self.add_mesh(i) for i in mesh]
        elif mesh.__class__.__name__ != 'Trimesh':
            return

        if 'name' in mesh.metadata: 
            name_mesh = mesh.metadata['name']
        else:
            name_mesh = 'mesh_' + str(len(self.meshes))

        self.meshes[name_mesh] = mesh

        if 'transforms' in mesh.metadata:
            transforms = np.array(mesh.metadata['transforms'])
        else:
            transforms = np.eye(4).reshape((-1,4,4))

        for i, transform in enumerate(transforms):
            name_node = name_mesh + '/' + str(i)
            self.instances[name_node] = name_mesh
            self.transforms.update(frame_to = name_node, 
                                   matrix   = transform)

    def set_camera(center_rotation, quaternion, zoom):
        pass
                                   
    def _naive(self):
        from copy import deepcopy
        from collections import deque
        result = deque()
        for node_id, mesh_id in self.instances.items():
            transform = self.transforms.get(node_id)
            current = deepcopy(self.meshes[mesh_id])
            current.transform(transform)
            result.append(current)
        return np.array(result)

    def show(self, block=True):
        from .viewer import SceneViewer
        SceneViewer(self, block=block)
