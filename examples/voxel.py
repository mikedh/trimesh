from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import inspect
import trimesh
from trimesh.exchange.binvox import load_binvox
from trimesh.voxel import creation
from trimesh import voxel as v


dir_current = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
# the absolute path for our reference models
dir_models = os.path.abspath(
    os.path.join(dir_current, '..', 'models'))


base_name = 'chair_model'
chair_mesh = trimesh.load(os.path.join(dir_models, '%s.obj' % base_name))
if isinstance(chair_mesh, (list, tuple)):
    chair_mesh = trimesh.util.concatenate([
        trimesh.Trimesh(mesh.vertices, mesh.faces) for mesh in chair_mesh])

binvox_path = os.path.join(dir_models, '%s.binvox' % base_name)
with open(binvox_path, "rb") as fp:
    chair_voxels = load_binvox(fp)

chair_voxels = v.Voxel(chair_voxels.encoding.dense, chair_voxels.transform)


def show(chair_mesh, chair_voxels, colors=(1, 1, 1, 0.3)):
    scene = chair_mesh.scene()
    scene.add_geometry(chair_voxels.as_boxes(colors=colors))
    scene.show()


print('binvox-loaded chair (red)')
show(chair_mesh, chair_voxels, colors=(1, 0, 0, 0.3))

voxelizer = creation.MeshVoxelizer(chair_mesh, np.max(chair_mesh.extents) / 32)
voxelized_chair_mesh = voxelizer.voxel_surface
print('voxelized chair (green)')
show(chair_mesh, voxelized_chair_mesh, colors=(0, 1, 0, 0.3))

transform = np.eye(4)
transform[:3] += np.random.normal(size=(3, 4)) * 0.2
transformed_chair_mesh = chair_mesh.copy().apply_transform(transform)
print('original transform volume: %s' % str(chair_voxels.transform.volume))
transformed_chair_voxels = chair_voxels.apply_transform(transform)
print('warped transform volume:   %s' % str(chair_voxels.transform.volume))
print('transformed loaded voxels (blue)')
show(transformed_chair_mesh, transformed_chair_voxels, colors=(0, 0, 1, 0.3))

shape = (50, 17, 63)
revox = chair_voxels.revoxelize(shape)
print('revoxelized (cyan)')
show(chair_mesh, revox, colors=(0, 1, 1, 0.3))

values = chair_voxels.encoding.dense.copy()
values[:values.shape[0] // 2] = 0
half_empty = v.Voxel(values, chair_voxels.transform)
stripped = half_empty.stripped
print('stripped (yellow)')
show(chair_mesh, stripped, colors=(1, 1, 0, 0.3))
