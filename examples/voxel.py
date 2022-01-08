from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import inspect
import trimesh

from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as v


dir_current = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
# the absolute path for our reference models
dir_models = os.path.abspath(
    os.path.join(dir_current, '..', 'models'))


def show(chair_mesh, chair_voxels, colors=(1, 1, 1, 0.3)):
    scene = chair_mesh.scene()
    scene.add_geometry(chair_voxels.as_boxes(colors=colors))
    scene.show()


if __name__ == '__main__':
    base_name = 'chair_model'
    chair_mesh = trimesh.load(os.path.join(dir_models, '%s.obj' % base_name))
    if isinstance(chair_mesh, trimesh.scene.Scene):
        chair_mesh = trimesh.util.concatenate([
            trimesh.Trimesh(mesh.vertices, mesh.faces)
            for mesh in chair_mesh.geometry.values()])

    binvox_path = os.path.join(dir_models, '%s.binvox' % base_name)
    chair_voxels = trimesh.load(binvox_path)

    chair_voxels = v.VoxelGrid(chair_voxels.encoding.dense, chair_voxels.transform)

    print('white: voxelized chair (binvox, exact)')
    show(chair_mesh, voxelize_mesh(chair_mesh, exact=True), colors=(1, 1, 1, 0.3))

    print('red: binvox-loaded chair')
    show(chair_mesh, chair_voxels, colors=(1, 0, 0, 0.3))

    voxelized_chair_mesh = chair_mesh.voxelized(np.max(chair_mesh.extents) / 32)
    print('green: voxelized chair (default).')
    show(chair_mesh, voxelized_chair_mesh, colors=(0, 1, 0, 0.3))

    shape = (50, 17, 63)
    revox = chair_voxels.revoxelized(shape)
    print('cyan: revoxelized.')
    show(chair_mesh, revox, colors=(0, 1, 1, 0.3))

    values = chair_voxels.encoding.dense.copy()
    values[:values.shape[0] // 2] = 0
    stripped = v.VoxelGrid(values, chair_voxels.transform.copy()).strip()
    print('yellow: stripped halved voxel grid. Transform is updated appropriately')
    show(chair_mesh, stripped, colors=(1, 1, 0, 0.3))

    transform = np.eye(4)
    transform[:3] += np.random.normal(size=(3, 4)) * 0.2
    transformed_chair_mesh = chair_mesh.copy().apply_transform(transform)
    print('original transform volume: %s'
          % str(chair_voxels.element_volume))

    chair_voxels.apply_transform(transform)
    print('warped transform volume:   %s' % str(chair_voxels.element_volume))
    print('blue: transformed voxels')
    print('Transformation is lazy, and each voxel is no longer a cube.')
    show(transformed_chair_mesh, chair_voxels, colors=(0, 0, 1, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='subdivide').fill()
    print('green: subdivided')
    show(chair_mesh, voxelized, colors=(0, 1, 0, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='ray')
    print('red: ray. Poor performance on thin structures')
    show(chair_mesh, voxelized, colors=(1, 0, 0, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='binvox')
    print('red: binvox (default). Poor performance on thin structures')
    show(chair_mesh, voxelized, colors=(1, 0, 0, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='binvox', wireframe=True)
    print('green: binvox (wireframe). Still doesn\'t capture all thin structures')
    show(chair_mesh, voxelized, colors=(0, 1, 0, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='binvox', exact=True)
    print('blue: binvox (exact). Does a good job')
    show(chair_mesh, voxelized, colors=(0, 0, 1, 0.3))

    voxelized = chair_mesh.voxelized(
        pitch=0.02,
        method='binvox',
        exact=True,
        downsample_factor=2,
        downsample_threshold=1)
    print('red: binvox (exact downsampled) surface')
    show(chair_mesh, voxelized, colors=(1, 0, 0, 0.3))

    chair_voxels = chair_mesh.voxelized(pitch=0.02, method='binvox', exact=True)

    voxelized = chair_voxels.copy().fill(method='base')
    print('blue: binvox (exact) filled (base). Gets a bit overly excited')
    show(chair_mesh, voxelized, colors=(0, 0, 1, 0.3))

    voxelized = chair_voxels.copy().fill(method='orthographic')
    print('green: binvox (exact) filled (orthographic).')
    print("Doesn't do much as should be expected")
    show(chair_mesh, voxelized, colors=(0, 1, 0, 0.3))
