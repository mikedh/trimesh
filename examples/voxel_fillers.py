from trimesh.scene import Scene
from trimesh.primitives import Sphere
from trimesh.voxel.morphology import fillers

mesh = Sphere()


def show(surface, filled, label):
    print(label)
    scene = Scene()
    scene.add_geometry(surface.as_boxes(colors=(1, 0, 0, 0.3)))
    scene.add_geometry(filled.as_boxes(colors=(0, 0, 1, 0.5)))
    scene.show()


# remove_internal produced unexpected results when boundary pixels are occupied
# not useful very often, but handy to demonstrate filling algorithms.
surface = mesh.voxelized(
    pitch=0.2, method='binvox', remove_internal=True)
for impl in fillers:
    show(surface, surface.copy().fill(method=impl), impl)


filled = mesh.voxelized(
    pitch=0.05, method='binvox', exact=True).fill(method='holes')
hollow = filled.copy().hollow()
print('filled volume, hollow_volume')
print(filled.volume, hollow.volume)
print('hollow voxel (zoom in to see hollowness)')
hollow.show()
