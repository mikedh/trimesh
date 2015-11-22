import numpy as np

from ..points          import transform_points
from ..util            import is_sequence
from ..transformations import rotation_matrix
from .transform_tree   import TransformTree

from collections import deque


class Scene:
    '''
    A simple scene graph which can be rendered directly via pyglet/openGL,
    or through other endpoints such as a raytracer. 

    Meshes and lights are added by name, which can then be moved by updating
    transform in the transform tree. 
    '''

    def __init__(self, 
                 node       = None, 
                 base_frame ='world'):

        # instance name : mesh name
        self.instances = {}

        # dictionary, mesh name : Trimesh object
        self.meshes     = {}
        self.flags      = {}
        self.camera     = None
        self.transforms = TransformTree(base_frame = base_frame)

        self.add_mesh(node)
        self.set_camera()

    def add_mesh(self, mesh):
        '''
        Add a mesh to the scene.

        If the mesh has multiple transforms defined in its metadata, 
        a new instance of the mesh will be created at each transform. 
        '''
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
            self.flags[name_node] = {'visible':True}
            self.transforms.update(frame_to = name_node, 
                                   matrix   = transform)

    @property
    def bounds(self):
        '''
        Return the overall bounding box of the scene.

        Returns
        --------
        bounds: (2,3) float points for min, max corner
        '''
        corners = deque()
        for instance, mesh_name in self.instances.items():
            transform = self.transforms.get(instance)
            corners.append(transform_points(self.meshes[mesh_name].bounds, 
                                            transform))
        corners = np.vstack(corners)
        bounds  = np.array([corners.min(axis=0), 
                            corners.max(axis=0)])
        return bounds

    @property
    def box_size(self):
        return np.diff(self.bounds, axis=0).reshape(-1)

    @property
    def scale(self):
        return self.box_size.max()

    @property
    def centroid(self):
        '''
        Return the center of the bounding box for the scene.

        Returns
        --------
        centroid: (3) float point for center of bounding box
        '''
        centroid = np.mean(self.bounds, axis=0)
        return centroid
            
    def set_camera(self, angles=None, distance=None, center=None):
        if center is None:
            center = self.centroid
        if distance is None:
            distance = np.diff(self.bounds, axis=0).max()
        if angles is None:
            angles = np.zeros(3)

        translation = np.eye(4)
        translation[0:3,3] = center
        translation[2][3] += distance*1.5

        transform = np.dot(rotation_matrix(angles[0], [1,0,0], point=center),
                           rotation_matrix(angles[1], [0,1,0], point=center))
        transform = np.dot(transform, translation)
        self.transforms.update(frame_from = 'camera', 
                               frame_to   = self.transforms.base_frame,
                               matrix     = transform)

    def dump(self):
        '''
        Append all meshes in scene to a list of meshes.
        '''
        from copy import deepcopy
        result = deque()
        for node_id, mesh_id in self.instances.items():
            transform = self.transforms.get(node_id)
            current = deepcopy(self.meshes[mesh_id])
            current.transform(transform)
            result.append(current)
        return np.array(result)

    def save_image(self, file_obj, resolution=(1024,768)):
        from .viewer import SceneViewer
        SceneViewer(self, save_image=file_obj, resolution=resolution)

    def show(self, **kwargs):
        from .viewer import SceneViewer
        SceneViewer(self, **kwargs)
 
