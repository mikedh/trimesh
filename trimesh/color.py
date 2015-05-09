import numpy as np
from colorsys import hsv_to_rgb

COLORS = {'red'    : [194,59,34],
          'purple' : [150,111,214],
          'blue'   : [119,158,203],
          'brown'  : [160,85,45]}
          
DEFAULT_COLOR  = COLORS['purple']

def random_color(dtype=np.uint8):
    golden_ratio_conjugate = 0.618033988749895
    h     = np.mod(np.random.random() + golden_ratio_conjugate, 1)
    color = np.array(hsv_to_rgb(h, 0.5, 0.95))
    if np.dtype(dtype).kind in 'iu':
        max_value = (2**(np.dtype(dtype).itemsize * 8)) - 1
        color    *= max_value
    color     = color.astype(dtype)
    return color
