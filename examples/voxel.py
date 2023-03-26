from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import inspect
import trimesh

from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as v

log = trimesh.util.log

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
    trimesh.util.attach_to_log()

    base_name = 'chair_model'
    chair_mesh = trimesh.load(os.path.join(dir_models, '%s.obj' % base_name))
    if isinstance(chair_mesh, trimesh.scene.Scene):
        chair_mesh = trimesh.util.concatenate([
            trimesh.Trimesh(mesh.vertices, mesh.faces)
            for mesh in chair_mesh.geometry.values()])

    binvox_path = os.path.join(dir_models, '%s.binvox' % base_name)
    chair_voxels = trimesh.load(binvox_path)

    chair_voxels = v.VoxelGrid(
        chair_voxels.encoding.dense,
        chair_voxels.transform)

    log.debug('white: voxelized chair (binvox, exact)')
    show(
        chair_mesh, voxelize_mesh(
            chair_mesh, exact=True), colors=(
            1, 1, 1, 0.3))

    log.debug('red: binvox-loaded chair')
    show(chair_mesh, chair_voxels, colors=(1, 0, 0, 0.3))

    voxelized_chair_mesh = chair_mesh.voxelized(
        np.max(chair_mesh.extents) / 32)
    log.debug('green: voxelized chair (default).')
    show(chair_mesh, voxelized_chair_mesh, colors=(0, 1, 0, 0.3))

    shape = (50, 17, 63)
    revox = chair_voxels.revoxelized(shape)
    log.debug('cyan: revoxelized.')
    show(chair_mesh, revox, colors=(0, 1, 1, 0.3))

    values = chair_voxels.encoding.dense.copy()
    values[:values.shape[0] // 2] = 0
    stripped = v.VoxelGrid(values, chair_voxels.transform.copy()).strip()
    log.debug(
        'yellow: stripped halved voxel grid. Transform is updated appropriately')
    show(chair_mesh, stripped, colors=(1, 1, 0, 0.3))

    transform = np.eye(4)
    transform[:3] += np.random.normal(size=(3, 4)) * 0.2
    transformed_chair_mesh = chair_mesh.copy().apply_transform(transform)
    log.debug('original transform volume: %s'
              % str(chair_voxels.element_volume))

    chair_voxels.apply_transform(transform)
    log.debug('warped transform volume:   %s' %
              str(chair_voxels.element_volume))
    log.debug('blue: transformed voxels')
    log.debug('Transformation is lazy, and each voxel is no longer a cube.')
    show(transformed_chair_mesh, chair_voxels, colors=(0, 0, 1, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='subdivide').fill()
    log.debug('green: subdivided')
    show(chair_mesh, voxelized, colors=(0, 1, 0, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='ray')
    log.debug('red: ray. Poor performance on thin structures')
    show(chair_mesh, voxelized, colors=(1, 0, 0, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='binvox')
    log.debug('red: binvox (default). Poor performance on thin structures')
    show(chair_mesh, voxelized, colors=(1, 0, 0, 0.3))

    voxelized = chair_mesh.voxelized(
        pitch=0.02, method='binvox', wireframe=True)
    log.debug(
        'green: binvox (wireframe). Still doesn\'t capture all thin structures')
    show(chair_mesh, voxelized, colors=(0, 1, 0, 0.3))

    voxelized = chair_mesh.voxelized(pitch=0.02, method='binvox', exact=True)
    log.debug('blue: binvox (exact). Does a good job')
    show(chair_mesh, voxelized, colors=(0, 0, 1, 0.3))

    voxelized = chair_mesh.voxelized(
        pitch=0.02,
        method='binvox',
        exact=True,
        downsample_factor=2,
        downsample_threshold=1)
    log.debug('red: binvox (exact downsampled) surface')
    show(chair_mesh, voxelized, colors=(1, 0, 0, 0.3))

    chair_voxels = chair_mesh.voxelized(
        pitch=0.02, method='binvox', exact=True)

    voxelized = chair_voxels.copy().fill(method='base')
    log.debug('blue: binvox (exact) filled (base). Gets a bit overly excited')
    show(chair_mesh, voxelized, colors=(0, 0, 1, 0.3))

    voxelized = chair_voxels.copy().fill(method='orthographic')
    log.debug('green: binvox (exact) filled (orthographic).')
    log.debug("Doesn't do much as should be expected")
    show(chair_mesh, voxelized, colors=(0, 1, 0, 0.3))
