import numpy as np

from .constants import tol, log

_fcl_exists = True
try:
    import fcl  # pip install python-fcl
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
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(b, t)

        # Add collision object to set
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
        self._objs[name] = {
            'obj': o,
            'geom': b
        }
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
            self._manager.unregisterObject(self._objs[name]['obj'])
            self._manager.update(self._objs[name]['obj'])
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
            o = self._objs[name]['obj']
            o.setRotation(transform[:3, :3])
            o.setTranslation(transform[:3, 3])
            self._manager.update(o)
        else:
            raise ValueError('{} not in collision manager!'.format(name))

    def in_collision_single(self, mesh, transform=None, return_names=False):
        '''
        Check a single object for collisions against all objects in the manager.

        Parameters
        ----------
        mesh:          Trimesh object, the geometry of the collision object
        transform:    (4,4) float,     homogenous transform matrix for the object
        return_names : bool,           If true, a set is returned containing the names
                                       of all objects in collision with the provided object

        Returns
        -------
        is_collision: bool, True if a collision occurs and False otherwise
        names: set of str,  The set of names of objects that collided with the provided one
        '''
        if transform is None:
            transform = np.eye(4)

        # Create FCL data
        b = fcl.BVHModel()
        b.beginModel(len(mesh.vertices), len(mesh.faces))
        b.addSubModel(mesh.vertices, mesh.faces)
        b.endModel()
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(b, t)

        # Collide with manager's objects
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True))

        self._manager.collide(o, cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        # If we want to return the objects that were collision, collect them.
        if return_names:
            objs_in_collision = set()
            for contact in cdata.result.contacts:
                cg = contact.o1
                if cg == b:
                    cg = contact.o2
                name = self._extract_name(cg)
                objs_in_collision.add(name)
            return result, objs_in_collision
        else:
            return result

    def in_collision_internal(self, return_names=False):
        '''
        Check if any pair of objects in the manager collide with one another.

        Parameters
        ----------
        return_names : bool
            If true, a set is returned containing the names
            of all pairs of objects in collision.

        Returns
        -------
        is_collision: bool,  True if a collision occured between any pair of objects
                             and False otherwise
        names: set of 2-tup, The set of pairwise collisions. Each tuple
                             contains two names in alphabetical order indicating
                             that the two correspoinding objects are in collision.
        '''
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True))

        self._manager.collide(cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        if return_names:
            objs_in_collision = set()
            for contact in cdata.result.contacts:
                name1, name2 = self._extract_name(contact.o1), self._extract_name(contact.o2)
                names = tuple(sorted((name1, name2)))
                objs_in_collision.add(names)
            return result, objs_in_collision
        else:
            return result

    def in_collision_other(self, other_manager, return_names=False):
        '''
        Check if any object from this manager collides with any object from another manager.

        Parameters
        ----------
        other_manager: CollisionManager, another collision manager object
        return_names:  bool,             If true, a set is returned containing the names
                                         of all pairs of objects in collision.

        Returns
        -------
        is_collision: bool,  True if a collision occured between any pair of objects
                             and False otherwise
        names: set of 2-tup, The set of pairwise collisions. Each tuple
                             contains two names (first from this manager,
                             second from the other_manager) indicating
                             that the two correspoinding objects are in collision.
        '''
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=100000, enable_contact=True))
        self._manager.collide(other_manager._manager,
                              cdata, fcl.defaultCollisionCallback)
        result = cdata.result.is_collision

        if return_names:
            objs_in_collision = set()
            for contact in cdata.result.contacts:
                name1, name2 = self._extract_name(contact.o1), other_manager._extract_name(contact.o2)
                if name1 is None:
                    name1, name2 = self._extract_name(contact.o2), other_manager._extract_name(contact.o1)
                objs_in_collision.add((name1, name2))
            return result, objs_in_collision
        else:
            return result

    def _extract_name(self, geom):
        """Retrieve the name of an object from the manager by its collision geometry,
        or return None if not found.
        """
        for obj_name in self._objs:
            if self._objs[obj_name]['geom'] == geom:
                return obj_name
        return None
