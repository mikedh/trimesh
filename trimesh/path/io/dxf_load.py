import numpy as np

from ..constants import log
from ..entities  import Line, Arc
from ...util     import is_binary_file

from collections import deque

def angles_to_threepoint(angles, center, radius, normal=[0,0,1]):
    if angles[1] < angles[0]: 
        angles[1] += np.pi*2
    angles = [angles[0], np.mean(angles), angles[1]]
    planar = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    points = planar + center
    return points
    
def load_dxf(file_obj):
    def convert_line(e_data):
        e = dict(e_data)
        entities.append(Line(len(vertices) + np.arange(2)))
        vertices.extend(np.array([[e['10'], e['20']],
                                  [e['11'], e['21']]], dtype=np.float))        
    def convert_circle(e_data):
        e = dict(e_data)
        R = float(e['40'])
        C = np.array([e['10'], e['20']]).astype(float)
        points = angles_to_threepoint([0, np.pi], C, R)
        entities.append(Arc(points=(len(vertices) + np.arange(3)), closed=True))
        vertices.extend(points)
    def convert_arc(e_data):
        e = dict(e_data)
        R = float(e['40'])
        C = np.array([e['10'], e['20']], dtype=np.float)
        A = np.radians(np.array([e['50'], e['51']], dtype=np.float))
        
        points = angles_to_threepoint(A, C[0:2], R)  
        entities.append(Arc(len(vertices) + np.arange(3), closed=False))
        vertices.extend(points)
    def convert_polyline(e_data):
        unique, inverse = np.unique(e_data[:,0], return_inverse=True)
        e = dict()
        for i, key in enumerate(unique):
            e[key] = e_data[:,1][inverse==i]
        lines = np.column_stack((e['10'], e['20'])).astype(np.float)
        for i in range(len(lines) - 1):
            entities.append(Line([i+len(vertices), i+len(vertices)+1]))
        vertices.extend(lines)
        
    if is_binary_file(file_obj): 
        raise TypeError("Binary DXF is unsupported!")
        
    f = file_obj.read().decode('utf-8').upper()
    f = f[ f.find('ENTITIES') + 8:]
    f = f[:f.find('ENDSEC')]
    
    blob       = np.array(f.split())
    inflection = np.nonzero(blob == '5')[0] - 2
    index_next = dict(np.column_stack((inflection[:-1], 
                                       inflection[1:])))
                                       
    loaders = {'LINE'       : convert_line,
               'LWPOLYLINE' : convert_polyline,
               'ARC'        : convert_arc,
               'CIRCLE'     : convert_circle}
    vertices = deque()
    entities = deque()
    
    for chunk in np.array_split(blob, inflection):
        if len(chunk) < 2: continue
        entity_type = chunk[1]
        if entity_type in loaders:
            e_data = chunk[0:len(chunk)-len(chunk)%2].reshape((-1,2))   
            loaders[entity_type](e_data)
        else:            
            log.debug('Entity type %s not supported', entity_type)
            
    result = {'vertices' : np.vstack(vertices).astype(np.float),
              'entities' : np.array(entities)}
              
    return result
            
