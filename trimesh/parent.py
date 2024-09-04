"""
parent.py
-------------

The base class for Trimesh, PointCloud, and Scene objects
"""

import abc

import numpy as np

from . import bounds, caching
from . import transformations as tf
from .caching import cache_decorator
from .constants import tol
from .typed import Any, ArrayLike, Dict, NDArray, Optional
from .util import ABC


class Geometry(ABC):
    """
    `Geometry` is the parent class for all geometry.

    By decorating a method with `abc.abstractmethod` it means
    the objects that inherit from `Geometry` MUST implement
    those methods.
    """

    # geometry should have a dict to store loose metadata
    metadata: Dict

    @property
    @abc.abstractmethod
    def bounds(self) -> NDArray[np.float64]:
        pass

    @property
    @abc.abstractmethod
    def extents(self) -> NDArray[np.float64]:
        pass

    @abc.abstractmethod
    def apply_transform(self, matrix: ArrayLike) -> Any:
        pass

    @property
    @abc.abstractmethod
    def is_empty(self) -> bool:
        pass

    def __hash__(self):
        """
        Get a hash of the current geometry.

        Returns
        ---------
        hash : int
          Hash of current graph and geometry.
        """
        return self._data.__hash__()  # type: ignore

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def show(self):
        pass

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def export(self, file_obj, file_type=None):
        pass

    def __repr__(self):
        """
        Print quick summary of the current geometry without
        computing properties.

        Returns
        -----------
        repr : str
          Human readable quick look at the geometry.
        """
        elements = []
        if hasattr(self, "vertices"):
            # for Trimesh and PointCloud
            elements.append(f"vertices.shape={self.vertices.shape}")
        if hasattr(self, "faces"):
            # for Trimesh
            elements.append(f"faces.shape={self.faces.shape}")
        if hasattr(self, "geometry") and isinstance(self.geometry, dict):
            # for Scene
            elements.append(f"len(geometry)={len(self.geometry)}")
        if "Voxel" in type(self).__name__:
            # for VoxelGrid objects
            elements.append(str(self.shape)[1:-1])
        if "file_name" in self.metadata:
            display = self.metadata["file_name"]
            elements.append(f"name=`{display}`")
        return "<trimesh.{}({})>".format(type(self).__name__, ", ".join(elements))

    def apply_translation(self, translation: ArrayLike):
        """
        Translate the current mesh.

        Parameters
        ----------
        translation : (3,) float
          Translation in XYZ
        """
        translation = np.asanyarray(translation, dtype=np.float64)
        if translation.shape == (2,):
            # create a planar matrix if we were passed a 2D offset
            return self.apply_transform(tf.planar_matrix(offset=translation))
        elif translation.shape != (3,):
            raise ValueError("Translation must be (3,) or (2,)!")

        # manually create a translation matrix
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        return self.apply_transform(matrix)

    def apply_scale(self, scaling):
        """
        Scale the mesh.

        Parameters
        ----------
        scaling : float or (3,) float
          Scale factor to apply to the mesh
        """
        matrix = tf.scale_and_translate(scale=scaling)
        # apply_transform will work nicely even on negative scales
        return self.apply_transform(matrix)

    def __radd__(self, other):
        """
        Concatenate the geometry allowing concatenation with
        built in `sum()` function:
          `sum(Iterable[trimesh.Trimesh])`

        Parameters
        ------------
        other : Geometry
          Geometry or 0

        Returns
        ----------
        concat : Geometry
          Geometry of combined result
        """

        if other == 0:
            # adding 0 to a geometry never makes sense
            return self
        # otherwise just use the regular add function
        return self.__add__(type(self)(other))

    @cache_decorator
    def scale(self) -> float:
        """
        A loosely specified "order of magnitude scale" for the
        geometry which always returns a value and can be used
        to make code more robust to large scaling differences.

        It returns the diagonal of the axis aligned bounding box
        or if anything is invalid or undefined, `1.0`.

        Returns
        ----------
        scale : float
          Approximate order of magnitude scale of the geometry.
        """
        # if geometry is empty return 1.0
        if self.extents is None:
            return 1.0

        # get the length of the AABB diagonal
        scale = float((self.extents**2).sum() ** 0.5)
        if scale < tol.zero:
            return 1.0

        return scale

    @property
    def units(self) -> Optional[str]:
        """
        Definition of units for the mesh.

        Returns
        ----------
        units : str
          Unit system mesh is in, or None if not defined
        """
        return self.metadata.get("units", None)

    @units.setter
    def units(self, value: str) -> None:
        """
        Define the units of the current mesh.
        """
        self.metadata["units"] = str(value).lower().strip()


class Geometry3D(Geometry):
    """
    The `Geometry3D` object is the parent object of geometry objects
    which are three dimensional, including Trimesh, PointCloud,
    and Scene objects.
    """

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

        return primitives.Box(transform=transform, extents=self.extents, mutable=False)

    @caching.cache_decorator
    def bounding_box_oriented(self):
        """
        An oriented bounding box for the current mesh.

        Returns
        ---------
        obb : trimesh.primitives.Box
          Box object with transform and extents defined
          representing the minimum volume oriented
          bounding box of the mesh
        """
        from . import bounds, primitives

        to_origin, extents = bounds.oriented_bounds(self)
        return primitives.Box(
            transform=np.linalg.inv(to_origin), extents=extents, mutable=False
        )

    @caching.cache_decorator
    def bounding_sphere(self):
        """
        A minimum volume bounding sphere for the current mesh.

        Note that the Sphere primitive returned has an unpadded
        exact `sphere_radius` so while the distance of every vertex
        of the current mesh from sphere_center will be less than
        sphere_radius, the faceted sphere primitive may not
        contain every vertex.

        Returns
        --------
        minball : trimesh.primitives.Sphere
          Sphere primitive containing current mesh
        """
        from . import nsphere, primitives

        center, radius = nsphere.minimum_nsphere(self)
        return primitives.Sphere(center=center, radius=radius, mutable=False)

    @caching.cache_decorator
    def bounding_cylinder(self):
        """
        A minimum volume bounding cylinder for the current mesh.

        Returns
        --------
        mincyl : trimesh.primitives.Cylinder
          Cylinder primitive containing current mesh
        """
        from . import bounds, primitives

        kwargs = bounds.minimum_cylinder(self)
        return primitives.Cylinder(mutable=False, **kwargs)

    @caching.cache_decorator
    def bounding_primitive(self):
        """
        The minimum volume primitive (box, sphere, or cylinder) that
        bounds the mesh.

        Returns
        ---------
        bounding_primitive : object
          Smallest primitive which bounds the mesh:
          trimesh.primitives.Sphere
          trimesh.primitives.Box
          trimesh.primitives.Cylinder
        """
        options = [
            self.bounding_box_oriented,
            self.bounding_sphere,
            self.bounding_cylinder,
        ]
        volume_min = np.argmin([i.volume for i in options])
        return options[volume_min]

    def apply_obb(self, **kwargs):
        """
        Apply the oriented bounding box transform to the current mesh.

        This will result in a mesh with an AABB centered at the
        origin and the same dimensions as the OBB.

        Parameters
        ------------
        kwargs
          Passed through to `bounds.oriented_bounds`

        Returns
        ----------
        matrix : (4, 4) float
          Transformation matrix that was applied
          to mesh to move it into OBB frame
        """
        # save the pre-transform volume
        if tol.strict and hasattr(self, "volume"):
            volume = self.volume

        # calculate the OBB passing keyword arguments through
        matrix, extents = bounds.oriented_bounds(self, **kwargs)
        # apply the transform
        self.apply_transform(matrix)

        if tol.strict:
            # obb transform should not have changed volume
            if hasattr(self, "volume") and getattr(self, "is_watertight", False):
                assert np.isclose(self.volume, volume)
            # overall extents should match what we expected
            assert np.allclose(self.extents, extents)

        return matrix
