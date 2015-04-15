from time import time as time_function

#when running multiprocess functions, how many processes?
PROCESS_COUNT = 3

#when merging vertices, what tolerance to use
#in an ideal world this would be the same as TOL_ZERO
#however different CAD packages export vertices with
#different levels of precision
TOL_MERGE     = 1e-6

#less than this is considered zero
TOL_ZERO      = 1e-12

#what is the maximum facet length that is converted to a circle
#if you didn't do this, octagons would be converted to circles
#in the simplification step
TOL_FACET  = .1875
# what's the maximum facet angle
# .2 radians ~= 15 degrees
TOL_FACET_ANGLE = .2

EXPORT_PRECISION = '.5f'

RES_LENGTH = .15
RES_ANGLE  = .26

import logging
log = logging.getLogger('path')
log.addHandler(logging.NullHandler())
