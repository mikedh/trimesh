"""
features.py
---------------

In trimesh.comparison, we arbitrarily threshold identifier values
at a certain number of significant figures.

This file permutates meshes around and observes how their identifier,
which is supposed to be pretty invariant to translation and tessellation
changes. We use this to generate the arbitrary sigfig thresholds.
"""

import numpy as np
import trimesh

import time
import json
import os

import collections
import logging

TOL_ZERO = 1e-12


def permutations(mesh,
                 function=lambda x: x.identifier,
                 displacement_max=1e-8,
                 count=1000,
                 subdivisions=2,
                 cutoff=3600):
    """
    Permutate a mesh, record the maximum it deviates from the original mesh
    and the resulting value of an identifier function.

    Parameters
    ----------
    mesh:     Trimesh object
    function:     function which takes a single mesh as an argument
                  and returns an (n,) float vector
    subdivisions: the maximum number of times to subdivide the mesh
    count: int, number of times to permutate each subdivision step

    Returns
    -----------
    identifiers: numpy array of identifiers
    """

    identifiers = []
    start = time.time()

    # do subdivisions
    divided = [mesh.copy()]
    for j in range(subdivisions - 1):
        divided.append(divided[-1].copy().subdivide())

    for i, displacement in enumerate(np.linspace(0.0,
                                                 displacement_max / mesh.scale,
                                                 count)):
        # get one of the subdivided meshes
        current = np.random.choice(divided).copy()

        if i > (count / 10):
            # run first bunch without tessellation permutation
            current = current.permutate.tessellation()
            # after the first few displace it a lot

        transformed = trimesh.permutate.transform(current)
        # noisy = trimesh.permutate.noise(transformed, displacement)

        identifier = function(transformed)
        identifiers.append(identifier)

        if (time.time() - start) > cutoff:
            print('bailing for time:{} count:{}'.format(time.time() - start,
                                                        i))
            return np.array(identifiers)

    return np.array(identifiers)


def get_meshes(path='../../../models', cutoff=None):
    """
    Get a list of single- body meshes to test identifiers on.

    Parameters
    ------------
    path:   str, location of models
    cutoff: int, number of meshes to stop loading at

    Returns
    ------------
    meshes: (n,) list of Trimesh objects
    """

    bodies = collections.deque()
    for file_name in os.listdir(path):
        try:
            mesh = trimesh.load(os.path.join(path, file_name))
            split = mesh.split()
            bodies.extend(split)
            if len(split) > 1:
                bodies.append(mesh)
        except BaseException:
            continue

        if cutoff is not None and len(bodies) > cutoff:
            return np.array(bodies)

    for i in range(100):
        cylinder = trimesh.creation.cylinder(
            radius=np.random.random() * 100,
            height=np.random.random() * 1000,
            sections=int(np.clip(np.random.random() * 720,
                                 20,
                                 720)))

        capsule = trimesh.creation.capsule(
            radius=np.random.random() * 100,
            height=np.random.random() * 1000,
            count=np.clip(np.random.random(2) * 720,
                          20,
                          720).astype(int))
        bodies.append(cylinder)
        bodies.append(capsule)
    for i in range(10):
        bodies.append(trimesh.creation.random_soup(
            int(np.clip(np.random.random() * 1000,
                        20,
                        1000))))
    bodies.append(trimesh.creation.icosphere())
    bodies.append(trimesh.creation.uv_sphere())
    bodies.append(trimesh.creation.icosahedron())

    return np.array(bodies)


def data_stats(data):
    data = np.asanyarray(data, dtype=np.float64)

    # mean identifier
    mean = data.mean(axis=0)
    # thresholdable percentile
    percent = np.abs(mean - np.abs(np.percentile(data, 99.999, axis=0)))

    return mean, percent


if __name__ == '__main__':
    trimesh.util.attach_to_log(level=logging.INFO)

    meshes = get_meshes()

    print('loaded meshes!')

    # we want the whole thing to last less than
    hours = 5
    cutoff = (hours * 3600) / len(meshes)
    cutoff = 30
    result = []
    running = []

    for i, m in enumerate(meshes):

        # calculate permutations
        identifier = permutations(m,
                                  count=1000,
                                  cutoff=cutoff)
        # get data
        mean, percent = data_stats(identifier)

        nz = np.logical_and(np.abs(mean) > TOL_ZERO,
                            np.abs(percent) > TOL_ZERO)

        r = np.ones_like(mean) * 10
        r[nz] = np.round(np.log10(np.abs(mean[nz] / percent[nz]))) - 1

        running.append(r)
        result.append({'mean': mean.tolist(),
                       'percent': percent.tolist()})

        print('\n\n{}/{}'.format(i, len(meshes) - 1))
        print('mean', mean)
        print('percent', percent)
        print('oom', mean / percent)
        print('curun', running[-1])
        print('minrun', np.min(running, axis=0))
        print('meanrun', np.mean(running, axis=0))

        # every loop dump everything
        # thrash- ey for sure but intermediate results are great
        name_out = 'res.json'
        with open(name_out, 'w') as file_obj:
            json.dump(result,
                      file_obj,
                      indent=4)
