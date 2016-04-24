import unittest
import inspect
import logging
import os
import trimesh

import numpy as np

dir_current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dir_models  = os.path.join(dir_current, '../models')

log = logging.getLogger('trimesh')
log.addHandler(logging.NullHandler())

