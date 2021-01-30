import numpy as np
from scipy.spatial import cKDTree
import trimesh, tempfile, os, fem, time

def hex2skin_quad(HexElements,HexNodes):
    """
    Extraction of QUAD surface mesh composed by the free faces of HEX8 mesh
    Parameters
    -------------
    HexNodes: (np.array) Node coordinates
    HexElements: (np.array) Node connectivity
    
    Returns
    ----------
    Quadlements: (np.array) Elements connectivity of quad mesh 
    """
    #HEX to QUAD
    QuadElements=np.zeros((len(HexElements),24), dtype=np.int64)
    ids=[0, 1, 5, 4, 0, 3, 2, 1, 0, 4, 7, 3, 1, 2, 6, 5, 2, 3, 7, 6, 4, 5, 6, 7]
    for i in range(len(ids)):
        QuadElements[:,i]=HexElements[:,ids[i]]
    QuadElements=QuadElements.reshape(-1,4)
    
    QuadElmCentroid=HexNodes[QuadElements].mean(1)
    tree=cKDTree(QuadElmCentroid)
    neighbors=tree.query(QuadElmCentroid,2)
    elm_size=fem.get_element_size(HexNodes,HexElements)
    ids=(neighbors[0][:,1]>elm_size/1000.)
    return QuadElements[ids]

def quad2tria(SkinQuads):
    """
    Breaks the surface QUAD mesh in a surface TRIA mesh
    Parameters
    -------------
    SkinQuads: (np.array) Surface mesh of quad elments

    Returns
    ----------
    SkinTria: (np.array) Surface mesh of tria elments
    """
    SkinTria=np.zeros((len(SkinQuads),6), dtype=np.int64)
    ids=[0, 1, 2, 2, 3, 0]
    for i in range(len(ids)):
        SkinTria[:,i]=SkinQuads[:,ids[i]]
    return SkinTria.reshape(-1,3)

def skin(HexNodes,HexElements,xprint,vol_fraction):
    """
    Extraction of TRIA surface mesh composed by the free
    faces of HEX8 mesh
    Parameters
    -------------
    HexNodes: (np.array) Node coordinates
    HexElements: (np.array) Node connectivity
    xprint: (np.array) Density map

    Returns
    ----------
        (Trimesh array) Surface mesh as a trimesh object
    """

    # Remove hexElements with xprint less than eps
    eps=min(0.3, vol_fraction/2.)
    HexElements=HexElements[xprint.T[0]>eps,:]

    #Transform to quad elements
    SkinQuads=hex2skin_quad(HexElements,HexNodes)

    # extract skin quads
    SkinTria=quad2tria(SkinQuads)

    # transform to Tria
    SkinTriangles=HexNodes[SkinTria]

    return trimesh.Trimesh(**trimesh.triangles.to_kwargs(SkinTriangles))

def shower(HexElements,HexNodes,xprint,vol_fraction):
    """
    Utility function to display a trimesh of a volume HEX8 mesh
    Parameters
    -------------
    HexNodes: (np.array) Node coordinates
    HexElements: (np.array) Node connectivity
    xprint: (np.array) Density map

    Returns
    ----------
        (Trimesh Scene)
    """
    skin(HexElements,HexNodes,xprint,vol_fraction).show()

def run_binvox(filename,grid_size):
    """
    Utility function to run binvox and obtain the voxel mesh in .msh format 
    Parameters
    -------------
    filename: (str) file path
    grid_size: (int) number of Elements in larger axis

    Returns
    ----------
        (None)
    """
    st="binvox "+filename+" -d "+str(int(grid_size))+" -t msh"
    os.system(st)

def voxelized_mesh(mesh,grid_size):
    """
    Utility function to voxelize the given surface mesh    
    Parameters
    -------------
    mesh: (trimesh array) surface mesh to be voxelized
    grid_size: (int) number of Elements in larger axis

    Returns
    ----------
    HexNodes: (np.array) Node coordinates
    HexElements: (np.array) Node connectivity    
    """
    #export mesh to stl
    mesh_file_stl = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    mesh_file_stl.close()
    mesh.export(mesh_file_stl.name)
    mesh_file_msh=mesh_file_stl.name.replace(".stl",".msh")

    #run binvox
    run_binvox(mesh_file_stl.name,grid_size)
    
    HexNodes, Elements=read_msh_hex_mesh(mesh_file_msh)

    return HexNodes, Elements, len(Elements)

def read_msh_hex_mesh(file_name):
    """
    Reads msh file
    Parameters
    -------------
    filename: (str) file path
    
    Returns
    ----------
    Nodes (np.array) cordinates xyz
    Elements (np.array) node conectivity
    """

    file2 = open(file_name,mode='r')
    alltxt = file2.read()
    file2.close()
    alltxt=alltxt.split("\n")

    i=-1
    while True:
        i+=1
        if "$Nodes" in alltxt[i]: 
            i+=1
            tot_nodes=int(alltxt[i])
            Nodes=np.zeros((tot_nodes,3))
            for j in range(tot_nodes):
                i+=1
                coords=alltxt[i].split(" ")
                Nodes[j,0]=float(coords[1])
                Nodes[j,1]=float(coords[2])
                Nodes[j,2]=float(coords[3])
        if "$Elements" in alltxt[i]:
            i+=1
            tot_elm=int(alltxt[i])
            Elements=np.zeros((tot_elm,8),dtype=int)
            for j in range(tot_elm):
                i+=1
                coords=alltxt[i].split(" ")
                for l in range(8):
                    Elements[j,l]=int(coords[5+l])
            break

    return Nodes, Elements


def load_gmsh(file_name, gmsh_args=None):
    """
    Returns a surface mesh from CAD model in Open Cascade
    Breap (.brep), Step (.stp or .step) and Iges formats
    Or returns a surface mesh from 3D volume mesh using gmsh.
    For a list of possible options to pass to GMSH, check:
    http://gmsh.info/doc/texinfo/gmsh.html
    An easy way to install the GMSH SDK is through the `gmsh-sdk`
    package on PyPi, which downloads and sets up gmsh:
        >>> pip install gmsh-sdk
    Parameters
    --------------
    file_name : str
      Location of the file to be imported
    gmsh_args : (n, 2) list
      List of (parameter, value) pairs to be passed to
      gmsh.option.setNumber
    max_element : float or None
      Maximum length of an element in the volume mesh
    Returns
    ------------
    mesh : trimesh.Trimesh
      Surface mesh of input geometry
    """
    # use STL as an intermediate format
    #from ..exchange.stl import load_stl
    # do import here to avoid very occasional segfaults
    import gmsh

    # start with default args for the meshing step
    # Mesh.Algorithm=2 MeshAdapt/Delaunay, there are others but they may include quads
    # With this planes are meshed using Delaunay and cylinders are meshed
    # using MeshAdapt
    args = [("Mesh.Algorithm", 2),
            ("Mesh.CharacteristicLengthFromCurvature", 1),
            ("Mesh.MinimumCirclePoints", 32)]
    # add passed argument tuples last so we can override defaults
    if gmsh_args is not None:
        args.extend(gmsh_args)

    # formats GMSH can load
    supported = ['.brep', '.stp', '.step', '.igs', '.iges',
                 '.bdf', '.msh', '.inp', '.diff', '.mesh']

    # check extensions to make sure it is supported format
    if file_name is not None:
        if not any(file_name.lower().endswith(e)
                   for e in supported):
            raise ValueError(
                'Supported formats are: BREP (.brep), STEP (.stp or .step), ' +
                'IGES (.igs or .iges), Nastran (.bdf), Gmsh (.msh), Abaqus (*.inp), ' +
                'Diffpack (*.diff), Inria Medit (*.mesh)')
    else:
        raise ValueError('No import since no file was provided!')

    # if we initialize with sys.argv it could be anything
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add('Surface_Mesh_Generation')
    gmsh.open(file_name)

    # create a temporary file for the results
    out_data = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    # windows gets mad if two processes try to open the same file
    out_data.close()

    # we have to mesh the surface as these are analytic BREP formats
    if any(file_name.lower().endswith(e)
           for e in ['.brep', '.stp', '.step', '.igs', '.iges']):
        gmsh.model.geo.synchronize()
        # loop through our numbered args which do things, stuff
        for arg in args:
            gmsh.option.setNumber(*arg)
        # generate the mesh
        gmsh.model.mesh.generate(2)
        # write to the temporary file
        gmsh.write(out_data.name)
    else:
        gmsh.plugin.run("NewView")
        gmsh.plugin.run("Skin")
        gmsh.view.write(1, out_data.name)

    # load the data from the temporary outfile
    #with open(out_data.name, 'rb') as f:
    kwargs = trimesh.load(out_data.name)

    return kwargs


def to_volume(mesh,
              file_name=None,
              max_element=None,
              mesher_id=1):
    """
    Convert a surface mesh to a 3D volume mesh generated by gmsh.
    An easy way to install the gmsh sdk is through the gmsh-sdk
    package on pypi, which downloads and sets up gmsh:
        pip install gmsh-sdk
    Algorithm details, although check gmsh docs for more information:
    The "Delaunay" algorithm is split into three separate steps.
    First, an initial mesh of the union of all the volumes in the model is performed,
    without inserting points in the volume. The surface mesh is then recovered using H.
    Si's boundary recovery algorithm Tetgen/BR. Then a three-dimensional version of the
    2D Delaunay algorithm described above is applied to insert points in the volume to
    respect the mesh size constraints.
    The Frontal" algorithm uses J. Schoeberl's Netgen algorithm.
    The "HXT" algorithm is a new efficient and parallel reimplementaton
    of the Delaunay algorithm.
    The "MMG3D" algorithm (experimental) allows to generate
    anisotropic tetrahedralizations
    Parameters
    --------------
    mesh : trimesh.Trimesh
      Surface mesh of input geometry
    file_name : str or None
      Location to save output, in .msh (gmsh) or .bdf (Nastran) format
    max_element : float or None
      Maximum length of an element in the volume mesh
    mesher_id : int
      3D unstructured algorithms:
      1: Delaunay, 4: Frontal, 7: MMG3D, 10: HXT
    Returns
    ------------
    data : None or bytes
      MSH data, only returned if file_name is None
    """
    # do import here to avoid very occasional segfaults
    import gmsh

    # checks mesher selection
    if mesher_id not in [1, 4, 7, 10]:
        raise ValueError('unavilable mesher selected!')
    else:
        mesher_id = int(mesher_id)

    # set max element length to a best guess if not specified
    if max_element is None:
        max_element = np.sqrt(np.mean(mesh.area_faces))

    if file_name is not None:
        # check extensions to make sure it is supported format
        if not any(file_name.lower().endswith(e)
                   for e in ['.bdf', '.msh', '.inp', '.diff', '.mesh']):
            raise ValueError(
                'Only Nastran (.bdf), Gmsh (.msh), Abaqus (*.inp), ' +
                'Diffpack (*.diff) and Inria Medit (*.mesh) formats ' +
                'are available!')

    # exports to disk for gmsh to read using a temp file
    mesh_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    mesh_file.close()
    mesh.export(mesh_file.name)

    # starts Gmsh Python API script
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add('Nastran_stl')

    gmsh.merge(mesh_file.name)
    dimtag = gmsh.model.getEntities()[0]
    dim = dimtag[0]
    tag = dimtag[1]

    surf_loop = gmsh.model.geo.addSurfaceLoop([tag])
    gmsh.model.geo.addVolume([surf_loop])
    gmsh.model.geo.synchronize()

    # We can then generate a 3D mesh...
    gmsh.option.setNumber("Mesh.Algorithm3D", mesher_id)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_element)
    gmsh.model.mesh.generate(3)

    dimtag2 = gmsh.model.getEntities()[1]
    dim2 = dimtag2[0]
    tag2 = dimtag2[1]
    p2 = gmsh.model.addPhysicalGroup(dim2, [tag2])
    gmsh.model.setPhysicalName(dim, p2, 'Nastran_bdf')

    data = None
    # if file name is None, return msh data using a tempfile
    if file_name is None:
        out_data = tempfile.NamedTemporaryFile(suffix='.msh', delete=False)
        # windows gets mad if two processes try to open the same file
        out_data.close()
        gmsh.write(out_data.name)
        with open(out_data.name, 'rb') as f:
            data = f.read()
    else:
        gmsh.write(file_name)

    # close up shop
    gmsh.finalize()

    return data