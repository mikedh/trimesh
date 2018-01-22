import numpy as np

import collections

from .constants import tol, log

_fcl_exists = True
try:
    import fcl  # pip install python-fcl
except BaseException:
    log.warning('No FCL -- collision checking will not work')
    _fcl_exists = False


class CollisionManager(object):
    """
    A mesh-mesh collision manager.
    """

    def __init__(self):
        """
        Initialize a mesh-mesh collision manager.
        """
        if not _fcl_exists:
            raise ValueError('No FCL Available!')
        # {name: {geom:, obj}}
        self._objs = {}
        # {id(bvh) : str, name}
        # unpopulated values will return None
        self._names = collections.defaultdict(lambda: None)

        # cache BVH objects
        # {mesh.md5(): fcl.BVHModel object}
        self._bvh = {}
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def add_object(self,
                   name,
                   mesh,
                   transform=None):
        """
        Add an object to the collision manager.

        If an object with the given name is already in the manager, replace it.

        Parameters
        ----------
        name:      str, an identifier for the object
        mesh:      Trimesh object, the geometry of the collision object
        transform: (4,4) float, homogenous transform matrix for the object
        """

        # if no transform passed, assume identity transform
        if transform is None:
            transform = np.eye(4)
        transform = np.asanyarray(transform, dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError('transform must be (4,4)!')

        # create or recall from cache BVH
        bvh = self._get_BVH(mesh)
        # create the FCL transform from (4,4) matrix
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(bvh, t)

        # Add collision object to set
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
        self._objs[name] = {'obj': o,
                            'geom': bvh}
        # store the name of the geometry
        self._names[id(bvh)] = name

        self._manager.registerObject(o)
        self._manager.update()
        return o

    def remove_object(self, name):
        """
        Delete an object from the collision manager.

        Parameters
        ----------
        name: str, the identifier for the object
        """
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name]['obj'])
            self._manager.update(self._objs[name]['obj'])
            # remove objects from _objs
            geom_id = id(self._objs.pop(name)['geom'])
            # remove names
            self._names.pop(geom_id)
        else:
            raise ValueError('{} not in collision manager!'.format(name))

    def set_transform(self, name, transform):
        """
        Set the transform for one of the manager's objects.
        This replaces the prior transform.

        Parameters
        ----------
        name:      str, an identifier for the object already in the manager
        transform: (4,4) float, a new homogenous transform matrix for the object
        """
        if name in self._objs:
            o = self._objs[name]['obj']
            o.setRotation(transform[:3, :3])
            o.setTranslation(transform[:3, 3])
            self._manager.update(o)
        else:
            raise ValueError('{} not in collision manager!'.format(name))

    def in_collision_single(self, mesh, transform=None, return_names=False):
        """
        Check a single object for collisions against all objects in the manager.

        Parameters
        ----------
        mesh:         Trimesh object, the geometry of the collision object
        transform:    (4,4) float,    homogenous transform matrix
        return_names: bool,           If true, a set is returned containing the names
                                      of all objects in collision with the object

        Returns
        -------
        is_collision: bool, True if a collision occurs and False otherwise
        names: set of str,  The set of names of objects that collided with the
                            provided one
        """
        if transform is None:
            transform = np.eye(4)

        # Create FCL data
        b = self._get_BVH(mesh)
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(b, t)

        # Collide with manager's objects
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000,
                enable_contact=True))

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
        """
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
        """
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000, enable_contact=True))

        self._manager.collide(cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        if return_names:
            objs_in_collision = set()
            for contact in cdata.result.contacts:
                name1, name2 = (self._extract_name(contact.o1),
                                self._extract_name(contact.o2))
                names = tuple(sorted((name1, name2)))
                objs_in_collision.add(names)
            return result, objs_in_collision
        else:
            return result

    def in_collision_other(self, other_manager, return_names=False):
        """
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
        """
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000,
                enable_contact=True))
        self._manager.collide(other_manager._manager,
                              cdata,
                              fcl.defaultCollisionCallback)
        result = cdata.result.is_collision

        if return_names:
            objs_in_collision = set()
            for contact in cdata.result.contacts:
                name1, name2 = (self._extract_name(contact.o1),
                                other_manager._extract_name(contact.o2))
                if name1 is None:
                    name1, name2 = (self._extract_name(contact.o2),
                                    other_manager._extract_name(contact.o1))
                objs_in_collision.add((name1, name2))
            return result, objs_in_collision
        else:
            return result

    def min_distance_single(self, mesh, transform=None, return_name=False):
        """
        Get the minimum distance between a single object and any object in the
        manager.

        Parameters
        ----------
        mesh:          Trimesh object, the geometry of the collision object
        transform:    (4,4) float,     homogenous transform matrix for the object
        return_names : bool,           If true, return name of the closest object

        Returns
        -------
        distance: float, Min distance between mesh and any object in the manager
        name: str,  The name of the object in the manager that was closest
        """
        if transform is None:
            transform = np.eye(4)

        # Create FCL data
        b = self._get_BVH(mesh)

        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
        o = fcl.CollisionObject(b, t)

        # Collide with manager's objects
        ddata = fcl.DistanceData()
        self._manager.distance(o, ddata, fcl.defaultDistanceCallback)
        distance = ddata.result.min_distance

        # If we want to return the objects that were collision, collect them.
        if return_name:
            cg = ddata.result.o1
            if cg == b:
                cg = ddata.result.o2
            name = self._extract_name(cg)

            return distance, name
        else:
            return distance

    def min_distance_internal(self, return_names=False):
        """
        Get the minimum distance between any pair of objects in the manager.

        Parameters
        ----------
        return_names : bool
            If true, a 2-tuple is returned containing the names of the closest objects.

        Returns
        -------
        distance: float, Min distance between any two managed objects
        names: (2,) str, The names of the closest objects
        """
        ddata = fcl.DistanceData()

        self._manager.distance(ddata, fcl.defaultDistanceCallback)

        distance = ddata.result.min_distance

        if return_names:
            names = tuple(sorted((self._extract_name(ddata.result.o1),
                                  self._extract_name(ddata.result.o2))))

            return distance, names
        else:
            return distance

    def min_distance_other(self, other_manager, return_names=False):
        """
        Get the minimum distance between any pair of objects, one in each manager.

        Parameters
        ----------
        other_manager: CollisionManager, another collision manager object
        return_names:  bool,             If true, a 2-tuple is returned containing
                                         the names of the closest objects.

        Returns
        -------
        distance: float,     The min distance between a pair of objects,
                             one from each manager.
        names: 2-tup of str, A 2-tuple containing two names (first from this manager,
                             second from the other_manager) indicating
                             the two closest objects.
        """
        ddata = fcl.DistanceData()

        self._manager.distance(other_manager._manager,
                               ddata,
                               fcl.defaultDistanceCallback)

        distance = ddata.result.min_distance

        if return_names:
            name1, name2 = (self._extract_name(ddata.result.o1),
                            other_manager._extract_name(ddata.result.o2))
            if name1 is None:
                name1, name2 = (self._extract_name(ddata.result.o2),
                                other_manager._extract_name(ddata.result.o1))
            return distance, (name1, name2)
        else:
            return distance

    def _get_BVH(self, mesh):
        """
        Get a BVH for a mesh.

        Parameters
        -------------
        mesh: Trimesh object

        Returns
        --------------
        bvh: fcl.BVHModel object
        """
        bvh = mesh_to_BVH(mesh)
        return bvh

    def _extract_name(self, geom):
        """
        Retrieve the name of an object from the manager by its
        CollisionObject, or return None if not found.

        Parameters
        -----------
        geom: CollisionObject, BVHModel
        """
        return self._names[id(geom)]


def mesh_to_BVH(mesh):
    """
    Create a BVHModel object from a Trimesh object

    Parameters
    -----------
    mesh: Trimesh object

    Returns
    ------------
    bvh: fcl.BVHModel object
    """
    bvh = fcl.BVHModel()
    bvh.beginModel(num_tris_=len(mesh.faces),
                   num_vertices_=len(mesh.vertices))
    bvh.addSubModel(verts=mesh.vertices,
                    triangles=mesh.faces)
    bvh.endModel()
    return bvh


def scene_to_collision(scene):
    """
    Create collision objects from a trimesh.Scene object.

    Parameters
    ------------
    scene: trimesh.Scene object

    Returns
    ------------
    manager: CollisionManager object
    objects: {node name: CollisionObject}
    """
    manager = CollisionManager()
    objects = {}
    for node in scene.graph.nodes_geometry:
        T, geometry = scene.graph[node]
        objects[node] = manager.add_object(name=node,
                                           mesh=scene.geometry[geometry],
                                           transform=T)
    return manager, objects
