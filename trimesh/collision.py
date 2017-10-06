import numpy as np

from .constants import tol, log

_fcl_exists = True
try:
    import fcl # pip install python-fcl
except:
    log.warning('No FCL -- collision checking will not work')
    _fcl_exists = False

class CollisionManager(object):

    def __init__(self):
        if not _fcl_exists:
            raise ValueError('No FCL Available!')
        self._objs = {}
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def add_object(self, name, mesh, transform=None):
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
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
            self._manager.update(self._objs[name])
            del self._objs[name]

    def set_transform(self, name, transform):
        if name in self._objs:
            o = self._objs[name]
            o.setRotation(transform[:3,:3])
            o.setTranslation(transform[:3,3])
            self._manager.update(o)

    def in_collision_single(self, mesh, transform=None):
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
        cdata = fcl.CollisionData()
        self._manager.collide(cdata, fcl.defaultCollisionCallback)
        return cdata.result.is_collision

    def in_collision_other(self, other_manager):
        cdata = fcl.CollisionData()
        self._manager.collide(other_manager._manager, cdata, fcl.defaultCollisionCallback)
        return cdata.result.is_collision

