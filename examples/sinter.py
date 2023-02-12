"""
A demo for packing a volume with multiple meshes as you
might for a powder volume in a sintered printing process.
"""
import os
import trimesh

import numpy as np

from trimesh.path import packing

from pyinstrument import Profiler


# path with our sample models
models = os.path.abspath(os.path.join(
    os.path.expanduser(os.path.dirname(__file__)), '..', 'models'))


def collect_meshes(count=None):
    """
    Collect single body watertight meshes from our
    models folder.

    Parameters
    ------------
    count : int or None
      If loaded more than this return.

    Returns
    -------------
    meshes : (n,) trimesh.Trimesh
      Loaded single body meshes.
    """
    meshes = []
    for file_name in sorted(os.listdir(models)):
        try:
            scene = trimesh.load(os.path.join(models, file_name),
                                 force='scene')
        except BaseException:
            pass
        for ori in scene.geometry.values():
            if (not isinstance(ori, trimesh.Trimesh) or
                    not ori.is_watertight or ori.volume < 0.001):
                continue

            # split into single body meshes
            for g in ori.split():
                g.visual = trimesh.visual.ColorVisuals(mesh=g)
                g.visual.face_colors = trimesh.visual.random_color()
                meshes.append(g)

        if count is not None and len(meshes) > count:
            return meshes[:count]

    return meshes


if __name__ == '__main__':

    # get some sample data
    meshes = collect_meshes(count=100)

    print('loaded {} meshes'.format(len(meshes)))

    # place the meshes into the volume
    with Profiler() as P:
        placed, transforms, consume = packing.meshes(
            meshes, spacing=0.05)
    P.print()

    # none of the placed meshes should have overlapping AABB
    assert not packing.bounds_overlap([i.bounds for i in placed])

    # show the packed result
    trimesh.Scene(placed).show()

    # concatenate them into a single mesh
    concat = trimesh.util.concatenate(placed)

    # slice the volume
    sections = concat.section_multiplane(
        plane_origin=concat.bounds[0],
        plane_normal=[0, 0, 1],
        heights=np.linspace(0.0, 10.0, 100))

    # show a cross section in the middle
    sections[50].show()
