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
                 nodes       = None, 
                 world_frame ='world'):

        self.meshes      = {}
        self.lights      = {}
        self.camera      = None
        self.transforms  = TransformTree()
        self.world_frame = world_frame

        self.add_nodes(nodes)

    def add_nodes(self, nodes):
        if nodes is None: return
        if is_sequence(nodes):
            result = [self.add_nodes(i) for i in nodes]
            return result
        node_type = nodes.__class__.__name__
    
        if node_type == 'Trimesh':
            # use or create a name the mesh will be referred to with
            if 'name' in nodes.metadata: mesh_name = nodes.metadata['name']
            elif hasattr(nodes, 'name'): mesh_name = nodes.name
            else:                        mesh_name = 'mesh_' + str(len(self.meshes))

            self.meshes[mesh_name] = nodes
            self.transforms.update(self.world_frame,
                                   mesh_name, 
                                   matrix=np.eye(4))
            return True
        return False

    def show(self, smooth=None):
        from .viewer import SceneViewer
        SceneViewer(self)
