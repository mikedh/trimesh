import trimesh
from trimesh.voxel.morphology import fillers


def show(surface, filled):
    """
    Display a colored example.
    """
    scene = trimesh.Scene([surface.as_boxes(colors=(1, 0, 0, 0.3)),
                           filled.as_boxes(colors=(0, 0, 1, 0.5))])
    scene.show()


if __name__ == '__main__':
    trimesh.util.attach_to_log()

    mesh = trimesh.primitives.Sphere()

    # remove_internal produced unexpected results when boundary pixels
    # are occupied not useful very often
    # but handy to demonstrate filling algorithms.
    surface = mesh.voxelized(
        pitch=0.2, method='binvox', remove_internal=True)
    for impl in fillers:
        trimesh.util.log.debug(impl)
        show(surface, surface.copy().fill(method=impl))

    filled = mesh.voxelized(
        pitch=0.05, method='binvox', exact=True).fill(method='holes')
    hollow = filled.copy().hollow()
    trimesh.util.log.debug('filled volume, hollow_volume')
    trimesh.util.log.debug(filled.volume, hollow.volume)
    trimesh.util.log.debug('hollow voxel (zoom in to see hollowness)')
    hollow.show()
