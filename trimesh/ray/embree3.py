"""
Ray queries using the pyembree package with the
API wrapped to match our native raytracer.
"""
import numpy as np

from .parent import RayParent

# bindings for embree3
import embree


def _add_mesh(mesh, slot, device, scene):
    """
    Add a Trimesh to an embree device and scene.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to be added
    slot : int
      The slot to add the mesh at in the scene
    device : embree.Device
      Embree device.
    scene : embree.Scene
      Embree scene.
    """
    slot = int(slot)
    geometry = device.make_geometry(
        embree.GeometryType.Triangle)

    vertex_buffer = geometry.set_new_buffer(
        embree.BufferType.Vertex,         # buf_type
        slot,                             # slot
        embree.Format.Float3,             # fmt
        3 * np.dtype('float32').itemsize,  # byte_stride
        len(mesh.vertices))               # item_count
    vertex_buffer[:] = mesh.vertices.astype(np.float32)[:]

    face_buffer = geometry.set_new_buffer(
        embree.BufferType.Index,       # buf_type
        slot,                             # slot
        embree.Format.Uint3,           # fmt
        3 * np.dtype('uint32').itemsize,  # byte_stride,
        len(mesh.faces))                # item count
    face_buffer[:] = mesh.faces.astype(np.uint32)[:]

    geometry.commit()
    scene.attach_geometry(geometry)
    geometry.release()
    scene.commit()


class RayMeshIntersector(RayParent):

    def __init__(self, geometry):
        """
        Do ray- mesh queries.

        Parameters
        -------------
        geometry : Trimesh object
          Mesh to do ray tests on
        scale_to_box : bool
          If true, will scale mesh to approximate
          unit cube to avoid problems with extreme
          large or small meshes.
        """

        #self.mesh = geometry

        self._device = embree.Device()
        self._scene = self._device.make_scene()

        # add the mesh to the embree scene
        _add_mesh(mesh=geometry,
                  slot=0,
                  device=self._device,
                  scene=self._scene)

    def __repr__(self):
        return 'embree3.RayMesh'

    def intersects_id(self,
                      origins,
                      vectors,
                      multiple_hits=True,
                      max_hits=20,
                      return_locations=False):
        # inherits docstring from parent
        origins = np.asanyarray(origins, dtype=np.float32)
        vectors = np.asanyarray(vectors, dtype=np.float32)

        assert origins.shape == vectors.shape

        rayhit = embree.RayHit1M(len(origins))
        rayhit.org[:] = origins
        rayhit.dir[:] = vectors

        rayhit.tnear[:] = 0
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

        context = embree.IntersectContext()
        self._scene.intersect1M(context, rayhit)

        # self._scene.release()
        # self._device.release()
        #from IPython import embed
        # embed()

        # make sure to copy all return values
        # otherwise things sure get segfaulty
        return np.array(rayhit.prim_id, dtype=np.int64)
