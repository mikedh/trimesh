import numpy as np
from scipy.spatial import cKDTree
import trimesh
import tempfile
import os
from topology import fem


def hex2skin_quad(HexElements, HexNodes):
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
    # HEX to QUAD
    QuadElements = np.zeros((len(HexElements), 24), dtype=np.int64)
    ids = [0, 1, 5, 4, 0, 3, 2, 1, 0, 4, 7, 3, 1, 2, 6, 5, 2, 3, 7, 6, 4, 5, 6, 7]
    for i in range(len(ids)):
        QuadElements[:, i] = HexElements[:, ids[i]]
    QuadElements = QuadElements.reshape(-1, 4)

    QuadElmCentroid = HexNodes[QuadElements].mean(1)
    tree = cKDTree(QuadElmCentroid)
    neighbors = tree.query(QuadElmCentroid, 2)
    elm_size = fem.get_element_size(HexNodes, HexElements)
    ids = neighbors[0][:, 1] > elm_size / 1000.0
    return QuadElements[ids]


def quad2tria(SkinQuads):
    """
    Breaks the surface QUAD mesh in a surface TRIA mesh
    Parameters
    -------------
    SkinQuads: (np.array) Surface mesh of quad elements

    Returns
    ----------
    SkinTria: (np.array) Surface mesh of tria elements
    """
    SkinTria = np.zeros((len(SkinQuads), 6), dtype=np.int64)
    ids = [0, 1, 2, 2, 3, 0]
    for i in range(len(ids)):
        SkinTria[:, i] = SkinQuads[:, ids[i]]
    return SkinTria.reshape(-1, 3)


def skin(HexNodes, HexElements, xprint, vol_fraction):
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
    eps = min(0.3, vol_fraction / 2.0)
    HexElements = HexElements[xprint.T[0] > eps, :]

    # Transform to quad elements
    SkinQuads = hex2skin_quad(HexElements, HexNodes)

    # extract skin quads
    SkinTria = quad2tria(SkinQuads)

    # transform to Tria
    SkinTriangles = HexNodes[SkinTria]

    return trimesh.Trimesh(**trimesh.triangles.to_kwargs(SkinTriangles))


def shower(HexElements, HexNodes, xprint, vol_fraction):
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
    skin(HexElements, HexNodes, xprint, vol_fraction).show()


def run_binvox(filename, grid_size):
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
    st = "binvox " + filename + " -d " + str(int(grid_size)) + " -t msh"
    os.system(st)


def voxelized_mesh(mesh, grid_size):
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
    # export mesh to stl
    mesh_file_stl = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    mesh_file_stl.close()
    mesh.export(mesh_file_stl.name)
    mesh_file_msh = mesh_file_stl.name.replace(".stl", ".msh")

    # run binvox
    run_binvox(mesh_file_stl.name, grid_size)

    HexNodes, Elements = read_msh_hex_mesh(mesh_file_msh)

    return HexNodes, Elements, len(Elements)


def read_msh_hex_mesh(file_name):
    """
    Reads msh file
    Parameters
    -------------
    filename: (str) file path

    Returns
    ----------
    Nodes (np.array) coordinates xyz
    Elements (np.array) node conectivity
    """

    file2 = open(file_name, mode="r")
    alltxt = file2.read()
    file2.close()
    alltxt = alltxt.split("\n")

    i = -1
    while True:
        i += 1
        if "$Nodes" in alltxt[i]:
            i += 1
            tot_nodes = int(alltxt[i])
            Nodes = np.zeros((tot_nodes, 3))
            for j in range(tot_nodes):
                i += 1
                coords = alltxt[i].split(" ")
                Nodes[j, 0] = float(coords[1])
                Nodes[j, 1] = float(coords[2])
                Nodes[j, 2] = float(coords[3])
        if "$Elements" in alltxt[i]:
            i += 1
            tot_elm = int(alltxt[i])
            Elements = np.zeros((tot_elm, 8), dtype=int)
            for j in range(tot_elm):
                i += 1
                coords = alltxt[i].split(" ")
                for l in range(8):
                    Elements[j, l] = int(coords[5 + l])
            break

    return Nodes, Elements
