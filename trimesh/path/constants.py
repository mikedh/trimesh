from time  import time as time_function

from ..constants import TOL_PLANAR

#when running multiprocess functions, how many processes?
PROCESS_COUNT = 3

#when merging vertices, what tolerance to use
#in an ideal world this would be the same as TOL_ZERO
#however different CAD packages export vertices with
#different levels of precision
TOL_MERGE        = 1e-5

#less than this is considered zero
TOL_ZERO      = 1e-12

#what is the maximum facet length that is converted to a circle
#if you didn't do this, octagons would be converted to circles
#in the simplification step
TOL_FACET_LENGTH  = .1875
# what's the maximum facet angle
# .2 radians ~= 15 degrees
TOL_FACET_ANGLE  = .2
TOL_RADIUS       = 1e-5
TOL_ASPECT       = .1
EXPORT_PRECISION = '.5f'

# target length of a section when discretizing curves
RES_LENGTH = .15
# target angle of a section when discretizing curves
RES_ANGLE  = .2

RES_MAX_SECTIONS = 150
RES_MIN_SECTIONS = 10
RES_TEST_FACTOR  = .05


import logging
log = logging.getLogger('path')
log.addHandler(logging.NullHandler())
