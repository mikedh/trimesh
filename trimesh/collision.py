import numpy as np

from .constants import tol, log

_fcl_exists = True
try:
    import fcl # pip install python-fcl
except:
    log.warning('No FCL -- collision checking will not work')
    _fcl_exists = False

class CollisionManager(object):
    '''
    A mesh-mesh collision manager.
    '''

    def __init__(self):
        '''
        Initialize a mesh-mesh collision manager.
        '''
        if not _fcl_exists:
            raise ValueError('No FCL Available!')
        self._objs = {}
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def add_object(self, name, mesh, transform=None):
        '''
        Add an object to the collision manager.
        If an object with the given name is already in the manager, replace it.

        Parameters
        ----------
        name:      str, an identifier for the object
        mesh:      Trimesh object, the geometry of the collision object
        transform: (4,4) float, homogenous transform matrix for the object
        '''
        if transform is None:
            transform = np.eye(4)

        # Create FCL data
        b = fcl.BVHModel()
        b.beginModel(len(mesh.vertices), len(mesh.faces))
        b.addSubModel(mesh.vertices, mesh.faces)
        b.endModel()
        t = fcl.Transform(transform[:3,:3], transform[:3,3])
        o = fcl.CollisionObject(b, t)

        # Add collision object to set
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
        self._objs[name] = o
        self._manager.registerObject(o)
        self._manager.update()

    def remove_object(self, name):
        '''
        Delete an object from the collision manager.

        Parameters
        ----------
        name: str, the identifier for the object
        '''
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
            self._manager.update(self._objs[name])
            del self._objs[name]
        else:
            raise ValueError('{} not in collision manager!'.format(name))

    def set_transform(self, name, transform):
        '''
        Set the transform for one of the manager's objects. This replaces the prior transform.

        Parameters
        ----------
        name:      str, an identifier for the object already in the manager
        transform: (4,4) float, a new homogenous transform matrix for the object
        '''
        if name in self._objs:
            o = self._objs[name]
            o.setRotation(transform[:3,:3])
            o.setTranslation(transform[:3,3])
            self._manager.update(o)
        else:
            raise ValueError('{} not in collision manager!'.format(name))

    def in_collision_single(self, mesh, transform=None):
        '''
        Check a single object for collisions against all objects in the manager.

        Parameters
        ----------
        mesh:      Trimesh object, the geometry of the collision object
        transform: (4,4) float, homogenous transform matrix for the object

        Returns
        -------
        is_collision: bool, True if a collision occurs and False otherwise
        '''
        if transform is None:
            transform = np.eye(4)

        # Create FCL data
        b = fcl.BVHModel()
        b.beginModel(len(mesh.vertices), len(mesh.faces))
        b.addSubModel(mesh.vertices, mesh.faces)
        b.endModel()
        t = fcl.Transform(transform[:3,:3], transform[:3,3])
        o = fcl.CollisionObject(b, t)

        # Collide with manager's objects
        cdata = fcl.CollisionData()
        self._manager.collide(o, cdata, fcl.defaultCollisionCallback)
        return cdata.result.is_collision

    def in_collision_internal(self):
        '''
        Check if any pair of objects in the manager collide with one another.

        Returns
        -------
        is_collision: bool, True if a collision occured between any pair of objects
                            and False otherwise
        '''
        cdata = fcl.CollisionData()
        self._manager.collide(cdata, fcl.defaultCollisionCallback)
        return cdata.result.is_collision

    def in_collision_other(self, other_manager):
        '''
        Check if any object from this manager collides with any object from another manager.

        Parameters
        ----------
        other_manager: CollisionManager, another collision manager object

        Returns
        -------
        is_collision: bool, True if a collision occured between any pair of objects
                            and False otherwise
        '''
        cdata = fcl.CollisionData()
        self._manager.collide(other_manager._manager, cdata, fcl.defaultCollisionCallback)
        return cdata.result.is_collision
