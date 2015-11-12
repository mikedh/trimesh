import numpy as np

from ..arc        import arc_center
from ...constants import log
from ...constants import res_path as res


def export_path(path, file_type, file_obj=None):
    '''
    Export a Path object to a file- like object, or to a filename

    Arguments
    ---------
    file_obj:  a filename string or a file-like object
    file_type: str representing file type (eg: 'svg')
    process:   boolean flag, whether to process the mesh on load

    Returns:
    mesh: a single Trimesh object, or a list of Trimesh objects, 
          depending on the file format. 
    
    '''
    if ((not hasattr(file_obj, 'read')) and 
        (not file_obj is None)):
        file_type = (str(file_obj).split('.')[-1]).lower()
        file_obj  = open(file_obj, 'wb')
    export = _path_exporters[file_type](path)
    return _write_export(export, file_obj)

def export_dict(path):
    export_entities = [e.to_dict() for e in path.entities]
    export_object   = {'entities' : export_entities,
                       'vertices' : path.vertices.tolist()}
    return export_object

def export_svg(drawing):
    '''
    Will turn a path drawing into an SVG path string. 

    'holes' will be in reverse order, so they can be rendered as holes by
    rendering libraries
    '''
    points = drawing.vertices.view(np.ndarray)
    def circle_to_svgpath(center, radius, reverse):
        radius_str = format(radius, res.export)
        path_str  = '  M' + format(center[0]-radius, res.export) + ',' 
        path_str += format(center[1], res.export)       
        path_str += 'a' + radius_str + ',' + radius_str  
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(2*radius, res.export) +  ',0'
        path_str += 'a' + radius_str + ',' + radius_str
        path_str += ',0,1,' + str(int(reverse)) + ','
        path_str += format(-2*radius, res.export) + ',0Z  '
        return path_str
    def svg_arc(arc, reverse):
        '''
        arc string: (rx ry x-axis-rotation large-arc-flag sweep-flag x y)+
        large-arc-flag: greater than 180 degrees
        sweep flag: direction (cw/ccw)
        '''
        vertices = points[arc.points[::((reverse*-2) + 1)]]        
        vertex_start, vertex_mid, vertex_end = vertices
        C, R, N, angle = arc_center(vertices)
        if arc.closed: return circle_to_svgpath(C, R, reverse)
        large_flag = str(int(angle > np.pi))
        sweep_flag = str(int(np.cross(vertex_mid-vertex_start, 
                                      vertex_end-vertex_start) > 0))
        R_ex = format(R, res.export)
        x_ex = format(vertex_end[0],res.export)
        y_ex = format(vertex_end [1],res.export)
        arc_str  = 'A' + R_ex + ',' + R_ex + ' 0 ' 
        arc_str += large_flag + ',' + sweep_flag + ' '
        arc_str += x_ex + ',' + y_ex
        return arc_str
    def svg_line(line, reverse):
        vertex_end = points[line.points[-(not reverse)]]
        x_ex = format(vertex_end[0], res.export) 
        y_ex = format(vertex_end[1], res.export) 
        line_str = 'L' + x_ex + ',' + y_ex
        return line_str
    def svg_moveto(vertex_id):
        x_ex = format(points[vertex_id][0], res.export) 
        y_ex = format(points[vertex_id][1], res.export) 
        move_str = 'M' + x_ex + ',' + y_ex
        return move_str
    def convert_path(path, reverse=False):
        path     = path[::(reverse*-2) + 1]
        path_str = svg_moveto(drawing.entities[path[0]].end_points[-reverse])
        for i, entity_id in enumerate(path):
            entity = drawing.entities[entity_id]
            e_type = entity.__class__.__name__
            try: 
                path_str += converters[e_type](entity, reverse)
            except KeyError:
                log.warn('%s entity not available for export!', e_type)
        path_str += 'Z'
        return path_str

    converters = {'Line'  : svg_line,
                  'Arc'   : svg_arc}
    path_str = ''
    for path_index, path in enumerate(drawing.paths):
        reverse   = not (path_index in drawing.root)
        path_str += convert_path(path, reverse)
    return path_str

def _write_export(export, file_obj=None):
    '''
    Write a string to a file.
    If file_obj isn't specified, return the string

    Arguments
    ---------
    export: a string of the export data
    file_obj: a file-like object or a filename
    '''

    if file_obj is None:             
        return export
    elif hasattr(file_obj, 'write'): 
        out_file = file_obj
    else: 
        out_file = open(file_obj, 'wb')
    out_file.write(export)
    out_file.close()
    return export

_path_exporters = {'svg'  : export_svg,
                   'dict' : export_dict}
