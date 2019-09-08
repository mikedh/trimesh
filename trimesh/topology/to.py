import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from topology import alm


def top_opt_vars_init(n, vol_fraction):
    """
    Search for neighbors using cKDTREE for fast Query.
    Parameters
    -------------
    n: (int) Number Elements
    vol_fraction: (float) Volume Fraction Constraint

    Returns
    ----------
    xval, xfilt, xphy, xprint: (np.array) Init of density Maps
    """
    xval = np.ones((n, 1)) * vol_fraction
    xfilt = np.ones((n, 1)) * vol_fraction
    xphy = np.ones((n, 1)) * vol_fraction
    xprint = np.ones((n, 1)) * vol_fraction

    return xval, xfilt, xphy, xprint


def neighbors_search(Centroid, radius):
    """
    Search for neighbors using cKDTREE for fast Query.
    Parameters
    -------------
    Centroid: (np.array) Elements Centroids
    Radius:   (float) size of filtering radius

    Returns
    ----------
    (np.array) list of elements per element
    """
    tree = cKDTree(Centroid)
    return tree.query_ball_point(Centroid, radius)


def laplacian_calculation(Centroid, Neighbors, radius):
    """
    Calculate a sparse matrix for filtering operations.
    Parameters
    -------------

    Returns
    ----------
    matrix : scipy.sparse.coo.coo_matrix
    Filtering operator
    """

    # stack neighbors to 1D arrays
    col = np.concatenate(Neighbors)
    row = np.concatenate([[i] * len(n) for i, n in enumerate(Neighbors)])

    # the distance from elements centroid to its neighbors
    ones = np.ones(3)
    norms = [
        (radius - np.sqrt(np.dot((Centroid[i] - Centroid[n]) ** 2, ones)))
        for i, n in enumerate(Neighbors)
    ]

    # normalize group and stack into single array
    data = np.concatenate([i / i.sum() for i in norms])

    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)), shape=[len(Centroid)] * 2)

    return matrix


def to_grad_calc(n, xprint, ese, penal, vol_fraction):
    """
    Calculate gradient of objective function and constraint
    Parameters
    -------------
    n: (int) Number Elements
    vol_fraction: (float) Volume Fraction Constraint
    ese: (np.array) Strain Energy
    penal: (float) Penalty constant of Topology Optimization (from 2 to 6)

    Returns
    ----------
    dc, dv: (np.float (n,1)) gradient of objective function and constraint
    """

    # Partial Derivatives
    dc = -penal / np.maximum(0.01, xprint) * ese
    dv = np.ones((n, 1), dtype=float) / n / vol_fraction

    return dc, dv


def to_grad_filt(
    n,
    xprint,
    xphy,
    xfilt,
    mfilter,
    support_region,
    reverse_region,
    fem_print_order,
    ese,
    dc,
    dv,
    penal,
    beta,
    eta,
    fta,
):
    """
    Filtering gradients of the objective and constraint functions
    in order to avoid checker board phenomena and impose minimum member size
    Parameters
    -------------
    n: (int) Number Elements
    xprint, xphy, xfilt: (np.array (n,1)) density map
    mfilter: (scipy sparse matrix (n,n)) filtering operator matrix
    support_region: (list) index of elements supporting element i
    reverse_region: (list) index of elements in which it is present in the support region
    fem_print_order: (list) list of elements supported by printing order
    ese: (np.array (n,1)) Strain Energy
    dc, dv: (np.float (n,1)) gradient of objective function and constraint
    penal: (float) Penalty constant of Topology Optimization (from 2 to 6)
    beta: (float) constant for heavisidade projection - Aggressiveness (i.e. 18)
    eta: (float) constant for heavisidade projection - Thresholding parameter (i.e. 0.5)
    fta: (bol) False Overhang constraint not considered
               True  Overhang constraint enforced

    Returns
    ----------
    xprint: (np.float (n,1)) density map
    dc, dv: (np.float (n,1)) filtered gradient of objective function and constraint
    """

    # Partial Derivatives of ALM Constraint
    if fta:
        xprint, dc, dv = alm.ALMfilter(
            support_region, reverse_region, fem_print_order, xphy, dc, dv
        )

    # Partial Derivative of Heavisidade Projection
    dx = (
        beta
        * (1.0 - (np.tanh(beta * (xfilt - eta))) ** 2.0)
        / (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)))
    )
    dc_aux = dc * dx
    dv_aux = dv * dx

    # Filtering Sensitivities
    dc = mfilter.dot(dc_aux)
    dv = mfilter.dot(dv_aux)

    return xprint, dc, dv


def filt_den(
    vol_fraction,
    xval,
    mfilter,
    support_region,
    reverse_region,
    fem_print_order,
    beta,
    eta,
    fta,
):
    """
    Filtering densities maps in order to avoid checker board phenomena
    and impose minimum member size

    Parameters
    -------------
    vol_fraction: (float) Volume Fraction Constraint
    xval: (np.array (n,1)) density map
    mfilter: (scipy sparse matrix (n,n)) filtering operator matrix
    support_region: (list) index of elements supporting element i
    reverse_region: (list) index of elements in which it is present in the support region
    fem_print_order: (list) list of elements supported by printing order
    beta: (float) constant for heavisidade projection - Aggressiveness (i.e. 18)
    eta: (float) constant for heavisidade projection - Thresholding parameter (i.e. 0.5)
    fta: (bol) False Overhang constraint not considered
               True  Overhang constraint enforced

    Returns
    ----------
    xprint, xphy, xfilt: (np.array (n,1)) density map
    Mn: (float) discriteness level
    """

    # Filtering
    xfilt = mfilter.dot(xval)

    # Projection
    xphy = (np.tanh(beta * eta) + np.tanh(beta * (xfilt - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    )

    # AM filter
    if fta:
        xprint = alm.ALMfilter(support_region, reverse_region, fem_print_order, xphy)
    else:
        xprint = xphy.copy()

    # Level of discriteness
    Mn = 4.0 * np.abs(xprint * (1 - xprint)).sum() / len(xprint)

    return xprint, xphy, xfilt, Mn
