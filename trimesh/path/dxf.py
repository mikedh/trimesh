'''
Forked from apparently abandoned 
https://code.google.com/p/dxf-reader

Which is GPLv2, so this file is probably that, instead of MIT like
the rest of the project.

http://www.autodesk.com/techpubs/autocad/acadr14/dxf/
'''


from .entities import Line, Arc
from collections import deque

# py3
try:                from cStringIO import StringIO
except ImportError: from io import BytesIO as StringIO

import numpy as np

import logging
log = logging.getLogger('vector')
log.addHandler(logging.NullHandler())

class DFACE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        DFACE.cnt += 1
class DSOLID:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        DSOLID.cnt += 1
        
class ACAD_PROXY_ENTITY:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        ACAD_PROXY_ENTITY.cnt += 1
            
def ARC(e, scale = 1, x_off = 0, y_off = 0):
    x = e.data["10"] * scale + x_off 
    y = e.data["20"] * scale + y_off
            
    r = e.data["40"] * scale
    start = e.data["50"]
    end = e.data["51"]
    
    return {'ARC': [[x,y], [r, start, end]]}
    
class ATTDEF:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        ATTDEF.cnt += 1
        
class ATTRIB:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        ATTRIB.cnt += 1
class BODY:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        BODY.cnt += 1
        
def CIRCLE(e, scale = 1, x_off = 0, y_off = 0):
    x = e.data["10"] * scale + x_off
    y = e.data["20"] * scale + y_off    
    r = e.data["40"] * scale 
    return {'CIRCLE': [[x,y], r]}
    
class DIMENSION:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        DIMENSION.cnt += 1
def ELLIPSE(e, scale = 1, x_off = 0, y_off = 0):
    print(e.data)
    return None
    
class HATCH:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        HATCH.cnt += 1
class IMAGE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        IMAGE.cnt += 1
class INSERT:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        self.e = e
        self.b_name = e.data["2"]


        if "10" in e.data:
            x_off = e.data["10"] * scale
        else:
            x_off = 0
            
        if "20" in e.data:
            y_off = e.data["20"] * scale
        else:
            y_off = 0

        
        if '41' in e.data:
            scale = e.data['41'] * scale
        INSERT.cnt += 1
        
        for entity in block.entities:
            f = funit[entity.type]
            if entity.type != "INSERT":
            #rotate if needed    
                if "50" in e.data:
                    alfa = radians(float(e.data["50"]))
                    for pnt in 10,11,12:
                        x_pnt = str(pnt)
                        y_pnt = str(pnt+10)
                        if x_pnt in entity.data:
                            if type(entity.data[x_pnt]) != list:
                                x = entity.data[x_pnt]
                                y = entity.data[y_pnt] 
                                entity.data[x_pnt] = x * cos(alfa) + y * sin(alfa)
                                entity.data[y_pnt] = -(-x * sin(alfa) + y * cos(alfa))
                            else:
                                pass
                    if entity.type == "ARC":
                        for a in "50","51":
                            entity.data[a] = entity.data[a] - degrees(alfa)
                i = f( entity, scale = scale, x_off = x_off, y_off = y_off)

            else:
                i = f( entity)  
            self.tag = self.b_name
        INSERT.cnt += 1

class LEADER:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        LEADER.cnt += 1
        
def LINE(e, scale = 1, x_off = 0, y_off = 0):
        x0 = e.data["10"] * scale + x_off
        y0 = e.data["20"] * scale + y_off
        x1 = e.data["11"] * scale + x_off
        y1 = e.data["21"] * scale + y_off
        
        return {'LINE':[[x0,y0], [x1,y1]]}
        
    
class LWPOLYLINE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        self.e = e

        arg = []
        for x,y in zip(e.data["10"], e.data["20"]):
            arg.append(x * scale + x_off)
            y = y * scale + y_off
        LWPOLYLINE.cnt += 1

class MLINE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
            MLINE.cnt += 1
class MTEXT:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        MTEXT.cnt += 1
class OLEFRAME:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        OLEFRAME.cnt += 1
class OLE2FRAME:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        OLE2FRAME.cnt += 1
        
class POINT:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        POINT.cnt += 1
class POLYLINE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        POLYLINE.cnt += 1
class RAY:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        RAY.cnt += 1
class REGION:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        REGION.cnt += 1
class SEQEND:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        SEQEND.cnt += 1
class SHAPE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        SHAPE.cnt += 1
class SOLID:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        SOLID.cnt += 1
class SPLINE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        SPLINE.cnt += 1
class TEXT:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):

            # core fields
            h = int(e.data["40"] * scale)
            x = e.data["10"] * scale
            y = e.data["20"] * scale + (h / 2)
            txt = e.data["1"]
            
            # optional fields
            if "72" in e.data:
                h_just = ("w", "", "e", "aligned", "middle", "fit")
                h_just = h_just[e.data["72"]]
            else:
                h_just = "w"
                
            if "73" in e.data:
                v_just = ("s","s", "", "n")
                v_just = v_just[e.data["73"]]
            else:
                v_just = ""

            if h_just == "aligned":
                print("alig")
                return
            if h_just == "middle":
                #x2 = e.data["11"]
                #y2 = e.data["21"]
                #h_scale =  (y2 * scale - y) / (len(txt) * h) 
                #h *= h_scale 
                #h = int(h)
                h_just = ""
                return
            if h_just == "fit":
                print("fit")
                return

            if v_just+h_just == "":
                just = "center"
            else:
                just = v_just+h_just

            if "41" in e.data:
                h /= e.data["41"]
                h = int(h)
            TEXT.cnt += 1
class TOLERANCE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        TOLERANCE.cnt += 1
class TRACE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        TRACE.cnt += 1
class VERTEX:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        VERTEX.cnt += 1
class VIEWPORT:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        VIEWPORT.cnt += 1
class XLINE:
    cnt = 0
    def __init__(self,  e, scale = 1, x_off = 0, y_off = 0):
        XLINE.cnt += 1
   
funit = dict({"3DFACE":DFACE,
               "3DSOLID":DSOLID,
               "ACAD_PROXY_ENTITY":ACAD_PROXY_ENTITY,
               "ARC":ARC,
               "ATTDEF":ATTDEF,
               "ATTRIB":ATTRIB,
               "BODY":BODY,
               "CIRCLE":CIRCLE,
               "DIMENSION":DIMENSION,
               "ELLIPSE":ELLIPSE,
               "HATCH":HATCH,
               "IMAGE":IMAGE,
               "INSERT":INSERT,
               "LEADER":LEADER,
               "LINE":LINE,
               "LWPOLYLINE":LWPOLYLINE,
               "MLINE":MLINE,
               "MTEXT":MTEXT,
               "OLEFRAME":OLEFRAME,
               "OLE2FRAME":OLE2FRAME,
               "POINT":POINT,
               "POLYLINE":POLYLINE,
               "RAY":RAY,
               "REGION":REGION,
               "SEQEND":SEQEND,
               "SHAPE":SHAPE,
               "SOLID":SOLID,
               "SPLINE":SPLINE,
               "TEXT":TEXT,
               "TOLERANCE":TOLERANCE,
               "TRACE":TRACE,
               "VERTEX":VERTEX,
               "VIEWPORT":VIEWPORT,
               "XLINE":XLINE})
               
strings = []
floats = []
ints = []

strings += list(range(0, 10))     #String (255 characters maximum; less for Unicode strings)
floats += list(range(10, 60))     #Double precision 3D point
ints += list(range(60, 80))       #16-bit integer value
ints += list(range(90,100))       #32-bit integer value
strings += [100]            #String (255 characters maximum; less for Unicode strings)
strings += [102]            #String (255 characters maximum; less for Unicode strings
strings += [105]            #String representing hexadecimal (hex) handle value
floats += list(range(140, 148))   #Double precision scalar floating-point value
ints += list(range(170, 176))     #16-bit integer value
ints += list(range(280, 290))     #8-bit integer value
strings += list(range(300, 310))  #Arbitrary text string
strings += list(range(310, 320))  #String representing hex value of binary chunk
strings += list(range(320, 330))  #String representing hex handle value
strings += list(range(330, 369))  #String representing hex object IDs
strings += [999]            #Comment (string)
strings += list(range(1000, 1010))#String (255 characters maximum; less for Unicode strings)
floats += list(range(1010, 1060)) #Floating-point value
ints += list(range(1060, 1071))   #16-bit integer value
ints += [1071]              #32-bit integer value

def read_int(data):
    return int(data)
def read_float(data):
    return float(data)
def read_string(data):
    return str(data)
def read_none(data):
    return None

funs = [read_none] * 1072
for i in strings:
    funs[i] = read_string
for i in floats:
    funs[i] = read_float
for i in ints:
    funs[i] = read_int

class Header:
    def __init__(self):
        self.variables = dict()
        self.last_var = None
    def new_var(self, kw):
        self.variables.update({kw: dict()})
        self.last_var = self.variables[kw]
    def new_val(self, val):
        self.last_var.update({ str(val[0]) : val[1] })

class Entity:
    def __init__(self, _type):
        self.type = _type
        self.data = dict()
    def update(self, value):
        key = str(value[0])
        val = value[1]
        if key in self.data:
            if type(self.data[key]) != list:
                self.data[key] = [self.data[key]]
            self.data[key].append(val)
        else:
            self.data.update({key:val})
  
class Entities:
    def __init__(self):
        self.entities = []
        self.last = None
    def new_entity(self, _type):
        e = Entity(_type)
        self.entities.append(e)
        self.last = e
    def update(self, value):
        self.last.update(value)

class Block:
    def __init__(self, master):
        self.master = master
        self.data = dict()
        self.entities = []
        self.le = None
    def new_entity(self, value):
        self.le = Entity(value)
        self.entities.append(self.le)
    def update(self, value):
        if self.le is None:
            val = str(value[0])
            self.data.update({val:value[1]})
            if val == "2":
                self.master.blocks[value[1]] = self
        else:
            self.le.update(value)

class Blocks:
    def __init__(self):
        self.blocks = dict()
        self.last_var = None
    def new_block(self):
        b = Block(self)
        #self.blocks.append(b)
        self.last_block = b
        self.last_var = b
    def new_entity(self, value):
        self.last_block.new_entity(value)
    def update(self, value):
        self.last_block.update(value)

def _raw_dxf_file(file_in):
    data = []
    Skip = True
    fd   = StringIO(file_in.read())

    for line in fd:
        group_code = int(line)
        value = next(fd).decode('utf-8')
        value = value.replace('\r', '')
        value = value.replace('\n', '')
        value = value.lstrip(' ')
        value = value.rstrip(' ')
        value = funs[group_code](value)
        if (value != "SECTION") and Skip:
            continue
        else:
            Skip = False
        data.append((group_code, value))
    fd.close()
    data = iter(data)
    g_code, value = None, None
    sections = dict()
    
    while value != "EOF":
        try: g_code, value = next(data)
        except: break
        if value == "SECTION":
            g_code, value = next(data)
            sections[value] = []
            while value != "ENDSEC":
                if value == "HEADER":
                    he = Header()
                    while True:
                        g_code, value = next(data)
                        if value == "ENDSEC":
                            break
                        elif g_code == 9:
                            he.new_var(value)
                        else:
                            he.new_val((g_code, value))

                elif value == "BLOCKS":
                    bl = Blocks()
                    while True:
                        g_code, value = next(data)
                        if value == "ENDSEC":
                            break
                        elif value == "ENDBLK":
                            continue
                        elif value == "BLOCK":
                            bl.new_block()
                        elif g_code == 0 and value != "BLOCK":
                            bl.new_entity(value)
                        else:
                            bl.update((g_code, value))

                elif value == "ENTITIES":
                    en = Entities()
                    while True:
                        g_code, value = next(data)
                        if value == "ENDSEC":
                            break
                        elif g_code == 0 and value != "ENDSEC":
                            en.new_entity(value)
                        else:
                            en.update((g_code, value))
                try: g_code, value = next(data)
                except: break
    if len(en.entities) == 0: 
        raise NameError('No entities loaded!')
    return en.entities

def angles_to_threepoint(angles, center, radius, normal=[0,0,1]):
    if angles[1] < angles[0]: angles[1] += np.pi*2
    angles = [angles[0], np.mean(angles), angles[1]]
    planar = np.column_stack((np.cos(angles), np.sin(angles)))*radius
    return planar + center

def detect_binary_file(file_obj):
    '''
    Returns True if file has non-ASCII characters (> 0x7F, or 127)
    Should work in both Python 2 and 3
    '''
    start  = file_obj.tell()
    fbytes = file_obj.read(1024)
    file_obj.seek(start)
    is_str = isinstance(fbytes, str)
    for fbyte in fbytes:
        if is_str: code = ord(fbyte)
        else:      code = fbyte
        if code > 127: return True
    return False

def dxf_to_vector(file_obj):
    def convert_line(en):
        entities.append(Line(len(vertices) + np.arange(2)))
        vertices.append([en.data['10'], en.data['20']]) #, en.data['30']])
        vertices.append([en.data['11'], en.data['21']]) #, en.data['31']])
    def convert_circle(en):
        C = [en.data['10'], en.data['20']] #, en.data['30']]
        R =  en.data['40']
        points = angles_to_threepoint([0, np.pi], C, R).tolist()
        entities.append(Arc(points=(len(vertices) + np.arange(3)), closed=True))
        vertices.extend(points)
    def convert_arc(en):
        angles = [np.radians(en.data['50']), np.radians(en.data['51'])]
        C = [en.data['10'], en.data['20']] #, en.data['30']]
        R = en.data['40']
        points = angles_to_threepoint(angles, C[0:2], R).tolist()   
        entities.append(Arc(len(vertices) + np.arange(3), closed=False))
        vertices.extend(points)
    def convert_polyline(en):
        lines = np.column_stack((en.data['10'], en.data['20']))#,en.data['30']))
        for i in range(len(lines) - 1):
            entities.append(Line([i+len(vertices), i+len(vertices)+1]))
        vertices.extend(lines)
    def convert_spline(en):
        #http://www.autodesk.com/techpubs/autocad/acad2000/dxf/spline_dxf_06.htm
        control = np.column_stack((en.data['10'], en.data['20']))
        flag    = en.data['70']
        for i in range(len(control) - 1):
            entities.append(vp.Line([i+len(vertices), i+len(vertices)+1]))
        vertices.extend(control)
        
    conversions = {'ARC'       : convert_arc, 
                   'LINE'      : convert_line,
                   'CIRCLE'    : convert_circle,
                   'LWPOLYLINE': convert_polyline}
                   #'SPLINE'    : convert_spline}

    if detect_binary_file(file_obj): 
        raise NameError("Only ASCII DXF is supported")
        
    entities = deque()
    vertices = deque()

    for entity in _raw_dxf_file(file_obj):
        if entity.type in conversions: conversions[entity.type](entity)
        else: log.debug('Entity type %s is unsupported.', str(entity.type))

    vertices = np.array(vertices)
    entities = np.array(entities)
    
    log.debug('Successfully loaded %i entities from DXF', len(entities))
    return entities, vertices
