"""
parent.py
-------------

The base class for Trimesh, PointCloud, and Scene objects
"""
import abc
import sys

import numpy as np

from . import caching

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Geometry(ABC):

    """Parent of geometry classes.

    The `Geometry` object is the parent object of geometry classes, including:
    Trimesh, PointCloud, and Scene objects.

    By decorating a method with `abc.abstractmethod` it just means the objects
    that inherit from `Geometry` MUST implement those methods.
    """

    @abc.abstractproperty
    def bounds(self):
        pass

    @abc.abstractproperty
    def extents(self):
        pass

    @abc.abstractmethod
    def apply_transform(self):
        pass

    @abc.abstractmethod
    def is_empty(self):
        pass

    def apply_translation(self, translation):
        """
        Translate the current mesh.

        Parameters
        ----------
        translation : (3,) float
          Translation in XYZ
        """
        translation = np.asanyarray(translation, dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError('Translation must be (3,)!')

        matrix = np.eye(4)
        matrix[:3, 3] = translation
        self.apply_transform(matrix)

    def apply_scale(self, scaling):
        """
        Scale the mesh equally on all axis.

        Parameters
        ----------
        scaling : float
          Scale factor to apply to the mesh
        """
        scaling = float(scaling)
        if not np.isfinite(scaling):
            raise ValueError('Scaling factor must be finite number!')

        matrix = np.eye(4)
        matrix[:3, :3] *= scaling
        # apply_transform will work nicely even on negative scales
        self.apply_transform(matrix)

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def show(self):
        pass

    @caching.cache_decorator
    def bounding_box(self):
        """
        An axis aligned bounding box for the current mesh.

        Returns
        ----------
        aabb : trimesh.primitives.Box
          Box object with transform and extents defined
          representing the axis aligned bounding box of the mesh
        """
        from . import primitives

        transform = np.eye(4)
        # translate to center of axis aligned bounds
        transform[:3, 3] = self.bounds.mean(axis=0)

        aabb = primitives.Box(transform=transform,
                              extents=self.extents,
                              mutable=False)
        return aabb

    @caching.cache_decorator
    def bounding_box_oriented(self):
        """
        An oriented bounding box for the current mesh.

        Returns
        ---------
        obb : trimesh.primitives.Box
          Box object with transform and extents defined
          representing the minimum volume oriented bounding box of the mesh
        """
        from . import primitives, bounds
        to_origin, extents = bounds.oriented_bounds(self)
        obb = primitives.Box(transform=np.linalg.inv(to_origin),
                             extents=extents,
                             mutable=False)
        return obb

    @caching.cache_decorator
    def bounding_sphere(self):
        """
        A minimum volume bounding sphere for the current mesh.

        Note that the Sphere primitive returned has an unpadded, exact
        sphere_radius so while the distance of every vertex of the current
        mesh from sphere_center will be less than sphere_radius, the faceted
        sphere primitive may not contain every vertex

        Returns
        --------
        minball: trimesh.primitives.Sphere
          Sphere primitive containing current mesh
        """
        from . import primitives, nsphere
        center, radius = nsphere.minimum_nsphere(self)
        minball = primitives.Sphere(center=center,
                                    radius=radius,
                                    mutable=False)
        return minball

    @caching.cache_decorator
    def bounding_cylinder(self):
        """
        A minimum volume bounding cylinder for the current mesh.

        Returns
        --------
        mincyl : trimesh.primitives.Cylinder
          Cylinder primitive containing current mesh
        """
        from . import primitives, bounds
        kwargs = bounds.minimum_cylinder(self)
        mincyl = primitives.Cylinder(mutable=False, **kwargs)
        return mincyl

    @caching.cache_decorator
    def bounding_primitive(self):
        """
        The minimum volume primitive (box, sphere, or cylinder) that
        bounds the mesh.

        Returns
        ---------
        bounding_primitive : trimesh.primitives.Sphere
                             trimesh.primitives.Box
                             trimesh.primitives.Cylinder
          Primitive which bounds the mesh with the smallest volume
        """
        options = [self.bounding_box_oriented,
                   self.bounding_sphere,
                   self.bounding_cylinder]
        volume_min = np.argmin([i.volume for i in options])
        bounding_primitive = options[volume_min]
        return bounding_primitive
