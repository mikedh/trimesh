import os
import trimesh
import numpy as np
from PIL import Image

from trimesh.geometry import align_vectors
from trimesh import util
from trimesh import transformations


resolution = 256
fov = 60.
path = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'bunny.ply'))

mesh = trimesh.load(path)
scene = mesh.scene()
camera = scene.camera

camera.fov = (fov,) * 2
camera.resolution = (resolution,) * 2
origin, rays = scene.camera_rays()
rays = rays.reshape((resolution, resolution, 3))
offset = mesh.vertices - origin

# dists is vertices projected onto central ray
dists = np.dot(offset, rays[rays.shape[0] // 2, rays.shape[1] // 2])
closest = np.min(dists)
farthest = np.max(dists)
z = np.linspace(closest, farthest, resolution)
print('z range: %f, %f' % (closest, farthest))

vox = mesh.voxelized(1./resolution, method='binvox')

coords = np.expand_dims(rays, axis=-2) * np.expand_dims(z, axis=-1)
coords += origin
frust_vox_dense = vox.is_filled(coords)
sil = np.any(frust_vox_dense, axis=-1)
sil = sil.T  # change to image ordering (y, x)

image = np.array(Image.open(trimesh.util.wrap_as_stream(
    scene.save_image(resolution=None))))
image = image[..., :3]


def vis():
    # separate function to delay plt import
    import matplotlib.pyplot as plt
    _, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(image)
    ax1.imshow(sil)
    sil_image = image.copy()
    sil_image[np.logical_not(sil)] = 0
    ax2.imshow(sil_image)
    plt.show()


vis()
