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
    
def multi_dict(pairs):
    result = dict()
    for k, v in pairs:
        if k in result: 
            result[k].append(v)
        else:
            result[k] = [v]
    return result

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
        e     = multi_dict(e_data)
        lines = np.column_stack((e['10'], e['20'])).astype(np.float)
        for i in range(len(lines) - 1):
            entities.append(Line([i+len(vertices), i+len(vertices)+1]))
        vertices.extend(lines)
    def convert_spline(e_data):
        # preliminary
        e      = multi_dict(e_data)
        points = np.column_stack((e['10'], e['20'])).astype(np.float)
        knots  = np.array(e['40']).astype(float)
        

    if is_binary_file(file_obj): 
        raise TypeError("Binary DXF is unsupported!")
        
    # in a DXF file, lines come in pairs, 
    # a group code then the next line is the value
    # we are removing all whitespace then splitting with the
    # splitlines function which uses the universal newline method
    raw  = file_obj.read().decode('utf-8').upper().replace(' ', '')
    # if this reshape fails, it means the DXF is malformed
    blob = np.array(unicode.splitlines(raw)).reshape((-1,2))
    
    # find the section which contains the entities in the DXF file
    endsec       = np.nonzero(blob[:,1] == 'ENDSEC')[0]
    entity_start = np.nonzero(blob[:,1] == 'ENTITIES')[0][0]
    entity_end   = endsec[np.searchsorted(endsec, entity_start)]

    entity_blob = blob[entity_start:entity_end]
    group_code  = np.array(entity_blob[:,0])
    group_check = np.logical_or(group_code == '0', 
                                group_code == '5')
    inflection = np.nonzero(np.logical_and(group_check[:-1], 
                                           group_check[:-1] == group_check[1:]))[0]
    loaders = {'LINE'       : convert_line,
               'LWPOLYLINE' : convert_polyline,
               'ARC'        : convert_arc,
               'CIRCLE'     : convert_circle}
               #'SPLINE'     : convert_spline}
    vertices = deque()
    entities = deque()
    
    for chunk in np.array_split(entity_blob, inflection):
        if len(chunk) > 2:
            entity_type = chunk[0][1]
            if entity_type in loaders:
                loaders[entity_type](chunk)
            else:
                log.debug('Entity type %s not supported', entity_type)
            
    result = {'vertices' : np.vstack(vertices).astype(np.float),
              'entities' : np.array(entities)}
              
    return result
            
