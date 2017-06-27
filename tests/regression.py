import generic as g

import time
import psutil
import timeit
import subprocess


def typical_application():
    # make sure we can load everything we think we can
    # while getting a list of meshes to run tests on
    meshes = g.get_meshes(raise_error=True)
    g.log.info('Running tests on %d meshes', len(meshes))

    for mesh in meshes:
        g.log.info('Testing %s', mesh.metadata['file_name'])
        assert len(mesh.faces) > 0
        assert len(mesh.vertices) > 0

        assert len(mesh.edges) > 0
        assert len(mesh.edges_unique) > 0
        assert len(mesh.edges_sorted) > 0
        assert len(mesh.edges_face) > 0
        assert isinstance(mesh.euler_number, int)

        mesh.process()

        if not mesh.is_watertight:
            continue

        assert len(mesh.facets) == len(mesh.facets_area)
        if len(mesh.facets) == 0:
            continue

        faces = mesh.facets[mesh.facets_area.argmax()]
        outline = mesh.outline(faces)
        smoothed = mesh.smoothed()

        assert mesh.volume > 0.0

        section = mesh.section(plane_normal=[0, 0, 1],
                               plane_origin=mesh.centroid)

        sample = mesh.sample(1000)
        even_sample = g.trimesh.sample.sample_surface_even(mesh, 100)
        assert sample.shape == (1000, 3)
        g.log.info('finished testing meshes')

        # make sure vertex kdtree and triangles rtree exist

        t = mesh.kdtree()
        assert hasattr(t, 'query')
        g.log.info('Creating triangles tree')
        r = mesh.triangles_tree()
        assert hasattr(r, 'intersection')
        g.log.info('Triangles tree ok')

        # some memory issues only show up when you copy the mesh a bunch
        # specifically, if you cache c- objects then deepcopy the mesh this
        # generally segfaults randomly
        copy_count = 20
        g.log.info('Attempting to copy mesh %d times', copy_count)
        for i in range(copy_count):
            copied = mesh.copy()
        g.log.info('Multiple copies done')
        assert g.np.allclose(copied.identifier,
                             mesh.identifier)
        assert isinstance(mesh.identifier_md5, str)


def establish_baseline(*args, counts=[390, 3820, 1710]):
    '''
    Try to establish a baseline of how fast a computer is so
    we can normalize our tests to be meaningful even on garbage CI VM's.

    Runs a simple dot product, cross product, and list comprehension
    with some multiplication and conditionals. 

    Counts have been tuned to each be approximatly 1.0 on a mid-2014 i7.


    Arguments
    ----------
    counts: (3,) int, number of iterations for each of the three tests

    Returns
    ---------
    times: (3,) float, seconds for each of the three tests
    '''

    setup = 'import numpy as np'

    # test a dot product with itself
    dot = '\n'.join(('a = np.arange(3*10**3,dtype=np.float64).reshape((-1,3))',
                     'b = np.dot(a, a.T)'))
    # test a cross product
    cross = '\n'.join(('a = np.arange(3*10**4,dtype=np.float64).reshape((-1,3))',
                       'b = np.cross(a, a[::-1])'))

    # try a list comprehension with some stuff in it
    loop = '[i * 3.14 for i in np.arange(10**3) if i % 7 == 0]'

    time_dot = min(timeit.repeat(dot,
                                 setup,
                                 number=counts[0]))
    time_cross = min(timeit.repeat(cross,
                                   setup,
                                   number=counts[1]))
    time_loop = min(timeit.repeat(loop,
                                  setup,
                                  number=counts[2]))
    times = g.np.array([time_dot,
                        time_cross,
                        time_loop])

    return times


def machine_info():
    info = {}

    freq = psutil.cpu_freq()
    info['cpu_count'] = psutil.cpu_count()
    info['cpu_freq'] = {'min': freq.min,
                        'max': freq.max,
                        'current': freq.current}

    return info


if __name__ == '__main__':

    #baseline = establish_baseline()

    '''
    file_names = ['ballA.off',
                  'cycloidal.ply',
                  'featuretype.STL',
                  'angle_block.STL',
                  'soup.stl',
                  'bun_zipper_res2.ply']

    setup = 'import generic as g;'
    setup += 'm=g.get_mesh(\'{}\');'

    # test a dot product with itself
    dot = '\n'.join(('a = np.arange(3*10**3,dtype=np.float64).reshape((-1,3))',
                     'b = np.dot(a, a.T)'))

    tests = {
        'section': 'r=m.section(plane_normal=[0,0,1],plane_origin=m.centroid)'}

    clear = '\n'.join(['\nn=m.face_normals',
                       'c=m.centroid',
                       'm._cache.clear()',
                       'm._cache[\'face_normals\']=n',
                       'm._cache[\'centroid\']=c'])

    tests = {'section': clear + 'r=m.section(plane_normal=[0,0,1],plane_origin=m.centroid)',
             'bounding_box_oriented': clear + 'r=m.bounding_box_oriented'}

    timings = g.collections.defaultdict(dict)

    iterations = 10
    repeats = 3
    for file_name in file_names:
        for test_name, test_string in tests.items():
            time_min = min(timeit.repeat(test_string,
                                         setup.format(file_name),
                                         repeat=repeats,
                                         number=iterations))
            timings[test_name][file_name] = time_min / iterations

    result = {}

    result['cpu_info'] = subprocess.check_output(['cat', '/proc/cpuinfo'])
    result['baseline'] = baseline.tolist()
    result['timestamp'] = time.time()
    result['timings'] = timings
    '''

    from multiprocessing import Pool

    p = Pool(2)

    tic = [time.time()]
    a = [establish_baseline() for i in range(4)]
    tic.append(time.time())

    b = list(p.imap_unordered(establish_baseline, range(4)))
    tic.append(time.time())
