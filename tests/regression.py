try:
    from . import generic as g
except BaseException:
    import generic as g

import timeit


def typical_application():
    # make sure we can load everything we think we can
    # while getting a list of meshes to run tests on
    meshes = g.get_meshes(raise_error=True)

    for mesh in meshes:
        g.log.info('Testing %s', mesh.metadata['file_name'])
        assert len(mesh.faces) > 0
        assert len(mesh.vertices) > 0

        assert len(mesh.edges) > 0
        assert len(mesh.edges_unique) > 0
        assert len(mesh.edges_sorted) > 0
        assert len(mesh.edges_face) > 0
        assert isinstance(mesh.euler_number, int)

        if not mesh.is_volume:
            continue

        assert len(mesh.facets) == len(mesh.facets_area)
        if len(mesh.facets) == 0:
            continue

        faces = mesh.facets[mesh.facets_area.argmax()]
        outline = mesh.outline(faces)  # NOQA
        smoothed = mesh.smoothed()  # NOQA

        assert mesh.volume > 0.0

        section = mesh.section(plane_normal=[0, 0, 1],  # NOQA
                               plane_origin=mesh.centroid)

        sample = mesh.sample(1000)
        assert sample.shape == (1000, 3)

        ident = mesh.identifier_md5
        assert len(ident) > 0


def establish_baseline(counts=None):
    """
    Try to establish a baseline of how fast a computer is so
    we can normalize our tests to be meaningful even on garbage CI VM's.

    Runs a simple dot product, cross product, and list comprehension
    with some multiplication and conditionals.

    Counts have been tuned to each be approximately 1.0 on a mid-2014 i7.


    Arguments
    ----------
    counts: (3,) int
      Number of iterations for each of the three tests

    Returns
    ----------------
    times : dict
      Minimum times for each test
    """
    if counts is None:
        counts = [390, 3820, 1710]
    setup = 'import numpy as np'

    # test a dot product with itself
    dot = '\n'.join(('a = np.arange(3*10**3,dtype=np.float64).reshape((-1,3))',
                     'b = np.dot(a, a.T)'))
    # test a cross product
    cross = '\n'.join(
        ('a = np.arange(3*10**4,dtype=np.float64).reshape((-1,3))',
         'b = np.cross(a, a[::-1])'))

    # try a list comprehension with some stuff in it
    loop = '[i * 3.14 for i in np.arange(10**3) if i % 7 == 0]'

    times = {}
    times['dot'] = min(timeit.repeat(dot,
                                     setup,
                                     number=counts[0]))
    times['cross'] = min(timeit.repeat(cross,
                                       setup,
                                       number=counts[1]))
    times['loop'] = min(timeit.repeat(loop,
                                      setup,
                                      number=counts[2]))

    return times


def machine_info():
    """
    Get information about the current machine

    Returns
    ------------
    info : dict
      Contains information about machine
    """
    import psutil
    info = {}
    info['cpu_count'] = psutil.cpu_count()

    return info


if __name__ == '__main__':

    info = machine_info()
    info['baseline'] = establish_baseline()

    import pyinstrument

    profiler = pyinstrument.Profiler()
    profiler.start()

    typical_application()

    profiler.stop()
    g.log.debug(profiler.output_text(unicode=True, color=True))
