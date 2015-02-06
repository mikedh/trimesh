from time import time as time_function
TOL_ZERO   = 1e-12
TOL_MERGE  = 1e-10
TOL_PLANAR = TOL_ZERO

import logging as _logging
_log = _logging.getLogger(__name__)
_log.addHandler(_logging.NullHandler())

def log_time(method):
    def timed(*args, **kwargs):
        tic    = time_function()
        result = method(*args, **kwargs)
        _log.debug('%s executed in %.4f seconds.',
                   method.__name__,
                   time_function()-tic)
        return result

    timed.__name__ = method.__name__
    timed.__doc__  = method.__doc__
    return timed
