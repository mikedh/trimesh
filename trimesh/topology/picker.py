import trimesh, time
import pyglet
from pyglet.gl import *
from trimesh.viewer.windowed import SceneViewer
from scipy.spatial import cKDTree
import numpy as np

class ClickViewer(SceneViewer):
    """
    Calculates the world coordinates of part based on the specified screen coordinates
    using the matrix unprojecting aproach.
    Parameters
    -------------
    scene: (mesh.scene()) trimesh scene

    Returns
    ----------
    self.bc: (list) list of coordinates of the picked points
    """
    def on_mouse_press(self, x, y, button, modifiers):
        # call the parent method in SceneViewer
        super(self.__class__, self).on_mouse_press(x, y, button, modifiers)

        # On Double click
        if hasattr(self, 'last_mouse_release'):
            if (x, y, button) == self.last_mouse_release[:-1]:
                """Same place, same button"""
                if time.time() - self.last_mouse_release[-1] < 0.8:
                    
                    # screen coordinates
                    x = int(x)
                    y = int(y) 
                    
                    # get point depth
                    z0 = (GLfloat *1)()
                    glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, z0)
                    
                    # unproject screen coordinates to world coordinates
                    pmat = (GLdouble * 16)()
                    mvmat = (GLdouble * 16)()
                    viewport = (GLint * 4)()
                    px = (GLdouble)()
                    py = (GLdouble)()
                    pz = (GLdouble)()
                    glGetIntegerv(GL_VIEWPORT, viewport)
                    glGetDoublev(GL_PROJECTION_MATRIX, pmat)
                    glGetDoublev(GL_MODELVIEW_MATRIX, mvmat)
                    gluUnProject(x, y, z0[0], mvmat, pmat, viewport, px, py, pz)                  
                    coord=[px.value, py.value, pz.value]
                    try:
                        self.bc += coord
                    except:
                        self.bc = coord
                    print("Vertex added")
                    
        
    def on_mouse_release(self, x, y, button, modifiers):

        x = int(x)
        y = int(y) #height - int(y)
    
        self.last_mouse_release = (x, y, button, time.time())

def vertex_picker(mesh,HexNodes,forces,clamp=True):
    """
    Converts screen coordinnates to world coordinates and retrieves the ID of the nearst node.

    Parameters
    -------------
    mesh: (trimesh.array) trimesh mesh
    HexNodes: (np.array) Coordinates of HEX8 mesh
    Forces: (np.array) Forces Components [[F1(x),F1(y),F1(z)],
                                          [F2(x),F2(y),F2(z)],
                                          [...] ]
    clamp: (Boolean): if True, the selected Nodes are clamped (constrained in the 3 directions),
                   otherwise the user is asked to pick the nodes to be constrained per direction

    Returns
    ----------
        (list) List of nodes to be clamped and loaded with a force
    """

    # Contruct KDtree for fast query
    tree=cKDTree(HexNodes)
    
    # Displacement Constraints
    if clamp==True:
        print("Double click on Nodes to be Clamped")
        coord=ClickViewer(mesh.scene()).bc
        if len(coord)==0:
            raise Exception("No vertices picked")
        coord = np.asarray(coord).reshape(-1,3)
        ids_bc=tree.query(coord)
        bc_u=[list(ids_bc[1])]*3
    else:
        print("Double click on Nodes to be constrained in X")
        coord=ClickViewer(mesh.scene()).bc
        coord1 = np.asarray(coord).reshape(-1,3)
        print("Double click on Nodes to be constrained in Y")
        coord=ClickViewer(mesh.scene()).bc
        coord2 = np.asarray(coord).reshape(-1,3)
        print("Double click on Nodes to be constrained in Z")
        coord=ClickViewer(mesh.scene()).bc
        coord3 = np.asarray(coord).reshape(-1,3)
        # Just to be sure there is no duplicate selection
        ids_bcx=np.unique(tree.query(coord1)[1])
        ids_bcy=np.unique(tree.query(coord2)[1])
        ids_bcz=np.unique(tree.query(coord3)[1])
        bc_u=[list(ids_bcx),list(ids_bcy),list(ids_bcz)]

    bc_f=[]
    for i in range(len(forces)):
        print("Double click on Node(s) of force ",i+1," location")
        coord=ClickViewer(mesh.scene()).bc
        coord = np.asarray(coord).reshape(-1,3)
        ids_ff=np.unique(tree.query(coord)[1])
        bc_f.append(list(ids_ff))

    return bc_u, bc_f