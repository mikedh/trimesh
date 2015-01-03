import numpy as np

COLORS = {'red'    : [194,59,34],
          'purple' : [150,111,214],
          'blue'   : [119,158,203],
          'brown'  : [160,85,45]}
          
DEFAULT_COLOR  = COLORS['purple']

def hsv_to_rgb(h, s, v):
    '''
    http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
    HSV values in [0..1]
    returns [r, g, b] values from 0 to 255
    '''
    h_i = int(h*6)
    f   = h*6 - h_i
    p = v * (1 - s)
    q = v * (1 - f*s)
    t = v * (1 - (1 - f) * s)
    if   h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
        
    return (np.array([r,g,b])*256).astype(int)

def random_color():
    # use golden ratio
    golden_ratio_conjugate = 0.618033988749895
    h = np.mod(np.random.random() + golden_ratio_conjugate, 1)
    color = hsv_to_rgb(h, 0.5, 0.95)
    #color = np.int_(np.random.random(3)*255)
    return color