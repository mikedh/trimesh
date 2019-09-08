import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, bicgstab


def get_element_size(Nodes, Elements):
    """
    Calculates the size of HEX8 element via Diagonal
    Parameters
    -------------
    Nodes: (np.array) Node coordinates
    Elements: (np.array) Node connectivity

    Returns
    ----------
        (float): edge size of element
    """
    elm_coords = Nodes[Elements][0]
    elm_vert = elm_coords.reshape(-1, 3)
    dist_vert = np.linalg.norm(elm_vert - elm_vert[0], axis=1)
    index = np.where(dist_vert == np.max(dist_vert))[0][0]
    return dist_vert[index] / 3 ** 0.5


def get_centroid(Nodes, Elements):
    """
    Calculates Elements centroid
    Parameters
    -------------
    Nodes (np.array) coordinates of nodes
    Elements (np.array) node connectivity

    Returns
    ----------
    (np.array) element's centroid
    """
    return Nodes[Elements].mean(1)


def ids_sparse_KG(Elements):
    """
    Computes Indexs of coo_matrix for assembling global matrix
    Parameters
    -------------
    Elements - Element node connectivity

    Returns
    ----------
    iK, jK - Indices of coo_matrix
    edofmat - matrix of dofs
    """
    edofmat = (Elements.flatten()[np.newaxis]).T.dot(3 * np.ones((1, 3), dtype=int))
    edofmat[:, 1] += 1
    edofmat[:, 2] += 2
    edofmat = edofmat.reshape((-1, 24))

    iK = np.kron(edofmat, np.ones((24, 1))).flatten()
    jK = np.kron(edofmat, np.ones((1, 24))).flatten()
    return iK, jK, edofmat


def fem_solve(
    E, nu, penal, xprint, Nodes, Elements, bc_u, bc_f, forces, Iterative_Solver
):
    """
    Solves the FEM problem considering a regular hex8 (cube) voxel mesh. For further
    details on the problem formulation consult [1]. The current implementation
    corresponds to a generalization of the article's algorithm using external mesher
    "binbox" for arbitrary designs volumes and precomputed sitiffness matrices.

    [1] - Liu, K. & Tovar, A. An efficient 3D topology optimization code written in
          Matlab. Struct Multidisc Optim (2014) 50: 1175.

    Parameters
    -------------
    Material Props:
        E: (float) Young Modulus,
        nu: (float) Poison ratio (i.e. 0.3 for many materials -> 0.<nu<0.5 )
    Numerical Constants:
        penal: (float) Penalty constant for the optimization
    Mesh:
        Nodes: (np.array) X, Y, Z coords
        Elements: (np.array) Node conectivity
        xprint: (np.array) relative density maps to be applied
    Boundary Conditions:
        bc_u: (np.array (n,3)) Displacement
              [:,0] - list all nodes constrained in direction X
              [:,1] - list all nodes constrained in direction Y
              [:,2] - list all nodes constrained in direction Z
        bc_f: (list(n,m)) Node id list to apply Forces.
              The n is the number of forces
              The m is the number of nodes associated to the force n
        Forces: (np.array(n,3)) Forces Components [[F1(x),F1(y),F1(z)],
                                                   [F2(x),F2(y),F2(z)],
                                                   [...]]
    Iterative_Solver - If the system of dofs is too large
        it can be set to True, if no suffucient memory is available
        for direct solver. But, takes much longer computational time

    Returns
    ----------
    ESE: (np.array (n,1)) Elements Strain Energy
    """

    # print("\n Compute HEX8 Elementary K matrix")
    KE = lk_H8(nu)
    aux_xprint = np.maximum(0.01, xprint.T[0])

    # print("\n Global Matrix assembly")
    n = len(Nodes)
    iK, jK, edofmat = ids_sparse_KG(Elements)
    kk = (KE.flatten()[np.newaxis]).T
    sK = (kk * ((aux_xprint) ** penal * E)).flatten(order="F")
    KG = coo_matrix((sK, (iK, jK)), shape=(3 * n, 3 * n)).tocsc()
    KG = (KG + KG.T) * 0.5

    # print("\n Boundary Conditions")
    # Forces Vector
    F = np.zeros((3 * n, 1))
    U = np.zeros((3 * n, 1))

    # Get dofs
    consdofs = np.concatenate(
        (
            np.asarray(bc_u[0], dtype=int) * 3,
            np.asarray(bc_u[1], dtype=int) * 3 + 1,
            np.asarray(bc_u[2], dtype=int) * 3 + 2,
        )
    )
    freedofs = np.setdiff1d(np.arange(3 * n), consdofs)

    # Apply Boundary conditions
    # Apply displacement boundary via reduce system of equations
    KG = KG[freedofs][:, freedofs]

    # Apply the inputed n force(s), per node per direction
    for i in range(len(forces)):
        for j in range(len(bc_f[i])):
            bc_fdof_x = np.asarray(bc_f[i][j], dtype=int) * 3
            bc_fdof_y = np.asarray(bc_f[i][j], dtype=int) * 3 + 1
            bc_fdof_z = np.asarray(bc_f[i][j], dtype=int) * 3 + 2
            F[bc_fdof_x] += forces[i][0]
            F[bc_fdof_y] += forces[i][1]
            F[bc_fdof_z] += forces[i][2]

    # print("\n Solve K U = F")
    # print("\n   Number of Nodes:    "+str(len(Nodes)))
    # print("\n   Number of Elements: "+str(len(Elements)))

    if not Iterative_Solver:
        U[freedofs, 0] = spsolve(KG, F[freedofs, 0])
    else:
        U[freedofs, 0] = bicgstab(KG, F[freedofs, 0])[0]

    # print("\n Computes Strain Energy")
    ese = (aux_xprint ** penal * E) * (
        (np.dot(U[edofmat].reshape(-1, 24), KE) * U[edofmat].reshape(-1, 24)).sum(1)
    )

    return ese.reshape(-1, 1)


def lk_H8(nu):
    """
    Computes the local stiffness matrix of HEX Element using a precomputed matrix based on

    Liu, K. & Tovar, A. An efficient 3D topology optimization code written in Matlab.
    Struct Multidisc Optim (2014) 50: 1175.

    Parameters
    -------------
    nu: (float) Poisson ratio

    Returns
    ----------
    KE: (np.array (24,24)) local stiffness Matrix for a HEX8 Element
    """
    A = np.array(
        [
            [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
            [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12],
        ]
    )
    k = 1.0 / 144 * A.T.dot(np.array([[1], [nu]]))
    k = np.insert(k.reshape(-1), 0, 0)

    K1 = np.array(
        [
            [k[1], k[2], k[2], k[3], k[5], k[5]],
            [k[2], k[1], k[2], k[4], k[6], k[7]],
            [k[2], k[2], k[1], k[4], k[7], k[6]],
            [k[3], k[4], k[4], k[1], k[8], k[8]],
            [k[5], k[6], k[7], k[8], k[1], k[2]],
            [k[5], k[7], k[6], k[8], k[2], k[1]],
        ]
    )

    K2 = np.array(
        [
            [k[9], k[8], k[12], k[6], k[4], k[7]],
            [k[8], k[9], k[12], k[5], k[3], k[5]],
            [k[10], k[10], k[13], k[7], k[4], k[6]],
            [k[6], k[5], k[11], k[9], k[2], k[10]],
            [k[4], k[3], k[5], k[2], k[9], k[12]],
            [k[11], k[4], k[6], k[12], k[10], k[13]],
        ]
    )

    K3 = np.array(
        [
            [k[6], k[7], k[4], k[9], k[12], k[8]],
            [k[7], k[6], k[4], k[10], k[13], k[10]],
            [k[5], k[5], k[3], k[8], k[12], k[9]],
            [k[9], k[10], k[2], k[6], k[11], k[5]],
            [k[12], k[13], k[10], k[11], k[6], k[4]],
            [k[2], k[12], k[9], k[4], k[5], k[3]],
        ]
    )

    K4 = np.array(
        [
            [k[14], k[11], k[11], k[13], k[10], k[10]],
            [k[11], k[14], k[11], k[12], k[9], k[8]],
            [k[11], k[11], k[14], k[12], k[8], k[9]],
            [k[13], k[12], k[12], k[14], k[7], k[7]],
            [k[10], k[9], k[8], k[7], k[14], k[11]],
            [k[10], k[8], k[9], k[7], k[11], k[14]],
        ]
    )

    K5 = np.array(
        [
            [k[1], k[2], k[8], k[3], k[5], k[4]],
            [k[2], k[1], k[8], k[4], k[6], k[11]],
            [k[8], k[8], k[1], k[5], k[11], k[6]],
            [k[3], k[4], k[5], k[1], k[8], k[2]],
            [k[5], k[6], k[11], k[8], k[1], k[8]],
            [k[4], k[11], k[6], k[2], k[8], k[1]],
        ]
    )

    K6 = np.array(
        [
            [k[14], k[11], k[7], k[13], k[10], k[12]],
            [k[11], k[14], k[7], k[12], k[9], k[2]],
            [k[7], k[7], k[14], k[10], k[2], k[9]],
            [k[13], k[12], k[10], k[14], k[7], k[11]],
            [k[10], k[9], k[2], k[7], k[14], k[7]],
            [k[12], k[2], k[9], k[11], k[7], k[14]],
        ]
    )

    a1 = np.concatenate((K1, K2, K3, K4), axis=1)
    a2 = np.concatenate((K2.T, K5, K6, K3.T), axis=1)
    a3 = np.concatenate((K3.T, K6, K5.T, K2.T), axis=1)
    a4 = np.concatenate((K4, K3, K2, K1.T), axis=1)
    KE = 1.0 / ((nu + 1) * (1 - 2 * nu)) * np.concatenate((a1, a2, a3, a4), axis=0)

    return KE
