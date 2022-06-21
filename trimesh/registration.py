"""
registration.py
---------------

Functions for registering (aligning) point clouds with meshes.
"""

import numpy as np
import scipy.sparse as sparse

from . import util
from . import bounds
from . import transformations

from .transformations import transform_points
from .points import PointCloud, plane_fit
from .proximity import closest_point
from .triangles import points_to_barycentric

try:
    from scipy.spatial import cKDTree
except BaseException as E:
    # wrapping just ImportError fails in some cases
    # will raise the error when someone tries to use KDtree
    from . import exceptions
    cKDTree = exceptions.closure(E)


def mesh_other(mesh,
               other,
               samples=500,
               scale=False,
               icp_first=10,
               icp_final=50):
    """
    Align a mesh with another mesh or a PointCloud using
    the principal axes of inertia as a starting point which
    is refined by iterative closest point.

    Parameters
    ------------
    mesh : trimesh.Trimesh object
      Mesh to align with other
    other : trimesh.Trimesh or (n, 3) float
      Mesh or points in space
    samples : int
      Number of samples from mesh surface to align
    scale : bool
      Allow scaling in transform
    icp_first : int
      How many ICP iterations for the 9 possible
      combinations of sign flippage
    icp_final : int
      How many ICP iterations for the closest
      candidate from the wider search

    Returns
    -----------
    mesh_to_other : (4, 4) float
      Transform to align mesh to the other object
    cost : float
      Average squared distance per point
    """

    def key_points(m, count):
        """
        Return a combination of mesh vertices and surface samples
        with vertices chosen by likelihood to be important
        to registation.
        """
        if len(m.vertices) < (count / 2):
            return np.vstack((
                m.vertices,
                m.sample(count - len(m.vertices))))
        else:
            return m.sample(count)

    if not util.is_instance_named(mesh, 'Trimesh'):
        raise ValueError('mesh must be Trimesh object!')

    inverse = True
    search = mesh
    # if both are meshes use the smaller one for searching
    if util.is_instance_named(other, 'Trimesh'):
        if len(mesh.vertices) > len(other.vertices):
            # do the expensive tree construction on the
            # smaller mesh and query the others points
            search = other
            inverse = False
            points = key_points(m=mesh, count=samples)
            points_mesh = mesh
        else:
            points_mesh = other
            points = key_points(m=other, count=samples)

        if points_mesh.is_volume:
            points_PIT = points_mesh.principal_inertia_transform
        else:
            points_PIT = points_mesh.bounding_box_oriented.principal_inertia_transform

    elif util.is_shape(other, (-1, 3)):
        # case where other is just points
        points = other
        points_PIT = bounds.oriented_bounds(points)[0]
    else:
        raise ValueError('other must be mesh or (n, 3) points!')

    # get the transform that aligns the search mesh principal
    # axes of inertia with the XYZ axis at the origin
    if search.is_volume:
        search_PIT = search.principal_inertia_transform
    else:
        search_PIT = search.bounding_box_oriented.principal_inertia_transform

    # transform that moves the principal axes of inertia
    # of the search mesh to be aligned with the best- guess
    # principal axes of the points
    search_to_points = np.dot(np.linalg.inv(points_PIT),
                              search_PIT)

    # permutations of cube rotations
    # the principal inertia transform has arbitrary sign
    # along the 3 major axis so try all combinations of
    # 180 degree rotations with a quick first ICP pass
    cubes = np.array([np.eye(4) * np.append(diag, 1)
                      for diag in [[1, 1, 1],
                                   [1, 1, -1],
                                   [1, -1, 1],
                                   [-1, 1, 1],
                                   [-1, -1, 1],
                                   [-1, 1, -1],
                                   [1, -1, -1],
                                   [-1, -1, -1]]])

    # loop through permutations and run iterative closest point
    costs = np.ones(len(cubes)) * np.inf
    transforms = [None] * len(cubes)
    centroid = search.centroid

    for i, flip in enumerate(cubes):
        # transform from points to search mesh
        # flipped around the centroid of search
        a_to_b = np.dot(
            transformations.transform_around(flip, centroid),
            np.linalg.inv(search_to_points))

        # run first pass ICP
        matrix, junk, cost = icp(a=points,
                                 b=search,
                                 initial=a_to_b,
                                 max_iterations=int(icp_first),
                                 scale=scale)

        # save transform and costs from ICP
        transforms[i] = matrix
        costs[i] = cost

    # run a final ICP refinement step
    matrix, junk, cost = icp(a=points,
                             b=search,
                             initial=transforms[np.argmin(costs)],
                             max_iterations=int(icp_final),
                             scale=scale)

    # convert to per- point distance average
    cost /= len(points)

    # we picked the smaller mesh to construct the tree
    # on so we may have calculated a transform backwards
    # to save computation, so just invert matrix here
    if inverse:
        mesh_to_other = np.linalg.inv(matrix)
    else:
        mesh_to_other = matrix

    return mesh_to_other, cost


def procrustes(a,
               b,
               reflection=True,
               translation=True,
               scale=True,
               return_cost=True):
    """
    Perform Procrustes' analysis subject to constraints. Finds the
    transformation T mapping a to b which minimizes the square sum
    distances between Ta and b, also called the cost.

    Parameters
    ----------
    a : (n,3) float
      List of points in space
    b : (n,3) float
      List of points in space
    reflection : bool
      If the transformation is allowed reflections
    translation : bool
      If the transformation is allowed translations
    scale : bool
      If the transformation is allowed scaling
    return_cost : bool
      Whether to return the cost and transformed a as well

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)) or not util.is_shape(b, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    if len(a) != len(b):
        raise ValueError('a and b must contain same number of points!')

    # Remove translation component
    if translation:
        acenter = a.mean(axis=0)
        bcenter = b.mean(axis=0)
    else:
        acenter = np.zeros(a.shape[1])
        bcenter = np.zeros(b.shape[1])

    # Remove scale component
    if scale:
        ascale = np.sqrt(((a - acenter)**2).sum() / len(a))
        bscale = np.sqrt(((b - bcenter)**2).sum() / len(b))
    else:
        ascale = 1
        bscale = 1

    # Use SVD to find optimal orthogonal matrix R
    # constrained to det(R) = 1 if necessary.
    u, s, vh = np.linalg.svd(
        np.dot(((b - bcenter) / bscale).T, ((a - acenter) / ascale)))

    if reflection:
        R = np.dot(u, vh)
    else:
        # no reflection allowed, so determinant must be 1.0
        R = np.dot(np.dot(u, np.diag(
            [1, 1, np.linalg.det(np.dot(u, vh))])), vh)

    # Compute our 4D transformation matrix encoding
    # a -> (R @ (a - acenter)/ascale) * bscale + bcenter
    #    = (bscale/ascale)R @ a + (bcenter - (bscale/ascale)R @ acenter)
    translation = bcenter - (bscale / ascale) * np.dot(R, acenter)
    matrix = np.hstack((bscale / ascale * R, translation.reshape(-1, 1)))
    matrix = np.vstack(
        (matrix, np.array([0.] * (a.shape[1]) + [1.]).reshape(1, -1)))

    if return_cost:
        transformed = transform_points(a, matrix)
        cost = ((b - transformed)**2).mean()
        return matrix, transformed, cost
    else:
        return matrix


def icp(a,
        b,
        initial=np.identity(4),
        threshold=1e-5,
        max_iterations=20,
        **kwargs):
    """
    Apply the iterative closest point algorithm to align a point cloud with
    another point cloud or mesh. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually).

    Parameters
    ----------
    a : (n,3) float
      List of points in space.
    b : (m,3) float or Trimesh
      List of points in space or mesh.
    initial : (4,4) float
      Initial transformation.
    threshold : float
      Stop when change in cost is less than threshold
    max_iterations : int
      Maximum number of iterations
    kwargs : dict
      Args to pass to procrustes

    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """

    a = np.asanyarray(a, dtype=np.float64)
    if not util.is_shape(a, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    is_mesh = util.is_instance_named(b, 'Trimesh')
    if not is_mesh:
        b = np.asanyarray(b, dtype=np.float64)
        if not util.is_shape(b, (-1, 3)):
            raise ValueError('points must be (n,3)!')
        btree = cKDTree(b)

    # transform a under initial_transformation
    a = transform_points(a, initial)
    total_matrix = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for n_iteration in range(max_iterations):
        # Closest point in b to each point in a
        if is_mesh:
            closest, distance, faces = b.nearest.on_surface(a)
        else:
            distances, ix = btree.query(a, 1)
            closest = b[ix]

        # align a with closest points
        matrix, transformed, cost = procrustes(a=a,
                                               b=closest,
                                               **kwargs)

        # update a with our new transformed points
        a = transformed
        total_matrix = np.dot(matrix, total_matrix)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return total_matrix, transformed, cost


def nricp(smesh,
          tmesh,
          source_landmarks=None,
          target_landmarks=None,
          steps=None,
          eps=0.0001,
          gamma=1,
          distance_treshold=0.1,
          return_records=False,
          use_faces=True,
          vertex_normals=True,
          nb_neighbors=8):
    """
    Implementation of "Amberg et al. 2007: Optimal Step Nonrigid ICP Algorithms
    for Surface Registration."
    Allows to register non-rigidly a mesh on another or on a point cloud.
    The core algorithm is explained at the end of page 3 of the paper.

    Parameters
    ----------
    smesh : Trimesh
        Source mesh containing both vertices and faces.
    tmesh : Trimesh or PointCloud
        Target mesh. It can contain no faces or be a PointCloud.
    source_landmarks : (n,) int or (n, 3) float or ((n,) int, (n, 3) float)
        n landmarks on the the source mesh.
        Either represented as vertex indices (n,) int or of 3d positions (n, 3).
        It can also be represented as a tuple of triangle indices and barycentric
        coordinates ((n,) int, (n, 3),).
    target_landmarks : (n,) int or (n, 3) float or ((n,) int, (n, 3) float)
        Same as source_landmarks
    steps : Core parameters of the algorithms
        Iterable of iterables (ws, wl, wn, max_iter,).
        ws is smoothness term, wl weights landmark importance, wn normal importance
        and max_iter is the maximum number of iterations per step.
    eps : float
        If the error decrease if inferior to this value, the current step ends.
    gamma : float
        Weight the translation part against the rotational/skew part.
        Recommended value : 1.
    distance_treshold : float
        Distance threshold to account for a vertex match or not.
    return_records : bool
        If True, also returns all the intermediate results. It can help debugging
        and tune the parameters to match a specific case.
    use_faces : bool
        If True and if target mesh has faces, use proximity.closest_point to find
        matching points. Else use scipy's cKDTree object.
    vertex_normals :
        If True and if target mesh hhas faces, interpolate the normals of the target
        mesh matching points.
        Else use face normals or estimated normals if target mesh has no faces.
    nb_neighbors :
        number of neighbors used for normal estimation. Only used if target mesh has
        no faces or if use_faces is False.

    Returns
    ----------
    result : Trimesh
        The source mesh registered non-rigidly onto the target mesh surface.
    records : List[Tuple((n, 3), (n,))]
        The vertex positions and error at each intermediate step iterations.
    """

    def _solve_system(M_kron_G, D, wd_vec, nearest, ws, nE, nV, Dl, Ul, wl):
        # Solve for Eq. 12
        U = wd_vec * nearest
        use_landmarks = Dl is not None and Ul is not None
        A_stack = [ws * M_kron_G, D.multiply(wd_vec)]
        B_shape = (4 * nE + nV, 3)
        if use_landmarks:
            A_stack.append(wl * Dl)
            B_shape = (4 * nE + nV + Ul.shape[0], 3)
        A = sparse.csr_matrix(sparse.vstack(A_stack))
        B = sparse.lil_matrix(B_shape, dtype=np.float32)
        B[4 * nE: (4 * nE + nV), :] = U
        if use_landmarks:
            B[4 * nE + nV: (4 * nE + nV + Ul.shape[0]), :] = Ul * wl
        X = sparse.linalg.spsolve(A.T * A, A.T * B).toarray()
        return X

    def _node_arc_incidence(mesh, do_weight):
        # Computes node-arc incidence matrix of mesh (Eq.10)
        nV = mesh.edges.max() + 1
        nE = len(mesh.edges)
        rows = np.repeat(np.arange(nE), 2)
        cols = mesh.edges.flatten()
        data = np.ones(2 * nE, np.float32)
        data[1::2] = -1
        if do_weight:
            edge_lengths = np.linalg.norm(mesh.vertices[mesh.edges[:, 0]] -
                                          mesh.vertices[mesh.edges[:, 1]], axis=-1)
            data *= np.repeat(1 / edge_lengths, 2)
        return sparse.coo_matrix((data, (rows, cols)), shape=(nE, nV))

    def _create_D(vertex_3d_data):
        # Create Data matrix (Eq. 8)
        nV = len(vertex_3d_data)
        rows = np.repeat(np.arange(nV), 4)
        cols = np.arange(4 * nV)
        data = np.concatenate((vertex_3d_data, np.ones((nV, 1))), axis=-1).flatten()
        return sparse.csr_matrix((data, (rows, cols)), shape=(nV, 4 * nV))

    def _create_X(nV):
        # Create Unknowns Matrix (Eq. 1)
        X_ = np.concatenate((np.eye(3), np.array([[0, 0, 0]])), axis=0)
        return np.tile(X_, (nV, 1))

    def _create_Dl_Ul(D,
                      smesh,
                      tmesh,
                      source_landmarks,
                      target_landmarks,
                      centroid,
                      scale):
        # Create landmark terms (Eq. 11)

        Dl, Ul = None, None
        wrong_landmark_format = ValueError('landmarks must be formatted as a np.ndarray '
                                           '(n, 3) float (3d positions) or (n,) int '
                                           '(vertex indices) or as a tuple of two '
                                           'np.ndarray (n,) int (face indices) and'
                                           ' (n, 3) float (barycentric coordinates)')

        if source_landmarks is None or target_landmarks is None:
            # If no landmarks are provided, return None for both
            return Dl, Ul

        # Retrieve target landmark 3d positions
        if isinstance(target_landmarks, tuple):
            target_tids, target_barys = target_landmarks
            target_points = np.einsum('ij,ijk->ik',
                                      target_barys,
                                      tmesh.vertices[tmesh.faces[target_tids]])
        elif isinstance(target_landmarks, np.ndarray):
            if target_landmarks.ndim == 2:
                target_points = (target_landmarks - centroid[None, :]) / scale
            elif target_landmarks.ndim == 1 and target_landmarks.dtype == int:
                target_points = tmesh.vertices[target_landmarks]
            else:
                raise wrong_landmark_format
        else:
            raise wrong_landmark_format

        if isinstance(source_landmarks, np.ndarray) and \
                source_landmarks.ndim == 2 or \
                isinstance(source_landmarks, tuple):
            # If source landmark are provided as barycentric coordinates :
            # (u, v, w) : barycentric coordinates
            # x1, x2, x3 : positions of triangle vertices
            # y : position of target landmark
            # what we want : u * x1 + v * x2 + w * x3 = y
            # => u * x1 = y - v * x2 + w * x3
            # => v * x2 = y - u * x1 + w * x3
            # => w * x3 = y - u * x1 + v * x2
            if isinstance(source_landmarks, np.ndarray):
                source_landmarks = (source_landmarks - centroid[None, :]) / scale
                _, _, source_tids = closest_point(smesh, source_landmarks)
                source_barys = points_to_barycentric(
                    smesh.vertices[smesh.faces[source_tids]],
                    source_landmarks)
            else:
                source_tids, source_barys = source_landmarks
            source_tri_vids = smesh.faces[source_tids]
            # u * x1, v * x2 and w * x3 combined
            Dl = D[source_tri_vids.flatten(), :]
            Dl.data *= source_barys.flatten().repeat(np.diff(Dl.indptr))
            x0 = smesh.vertices[source_tri_vids[:, 0]]
            x1 = smesh.vertices[source_tri_vids[:, 1]]
            x2 = smesh.vertices[source_tri_vids[:, 2]]
            Ul0 = target_points - x1 * source_barys[:, 1, None] \
                - x2 * source_barys[:, 2, None]
            Ul1 = target_points - x0 * source_barys[:, 0, None] \
                - x2 * source_barys[:, 2, None]
            Ul2 = target_points - x0 * source_barys[:, 0, None] \
                - x1 * source_barys[:, 1, None]
            Ul = np.zeros((Ul0.shape[0] * 3, 3))
            Ul[0::3] = Ul0  # y - v * x2 + w * x3
            Ul[1::3] = Ul1  # y - u * x1 + w * x3
            Ul[2::3] = Ul2  # y - u * x1 + v * x2
        elif isinstance(source_landmarks, np.ndarray) and \
                source_landmarks.ndim == 1 and \
                source_landmarks.dtype == int:
            # Else if source landmarks are vertex indices, it is much simpler
            Dl = D[source_landmarks, :]
            Ul = target_points
        else:
            raise wrong_landmark_format
        return Dl, Ul

    # Number of edges and vertices in smesh
    nE = len(smesh.edges)
    nV = len(smesh.vertices)
    # Check if registration target is a mesh or a point cloud
    is_target_pc = isinstance(tmesh, PointCloud) or len(tmesh.faces) == 0

    # Center and unitize source and target
    centroid, scale = smesh.centroid, smesh.scale
    smesh.vertices = (smesh.vertices - centroid[None, :]) / scale
    tmesh.vertices = (tmesh.vertices - centroid[None, :]) / scale

    # Check whether using faces or points for matching point queries
    use_faces = use_faces and not is_target_pc
    if not use_faces:
        # Form KDTree if using points
        kdtree = cKDTree(tmesh.vertices)

    # Initialize transformed vertices
    transformed_vertices = smesh.vertices.copy()
    # Node-arc incidence (M in Eq. 10)
    M = _node_arc_incidence(smesh, True)
    # G (Eq. 10)
    G = np.diag([1, 1, 1, gamma])
    # M kronecker G (Eq. 10)
    M_kron_G = sparse.kron(M, G)
    # D (Eq. 8)
    D = _create_D(smesh.vertices)
    # D but for normal computation from the transformations X
    DN = _create_D(smesh.vertex_normals)
    # Unknowns 4x3 transformations X (Eq. 1)
    X = _create_X(nV)
    # Landmark related terms (Eq. 11)
    Dl, Ul = _create_Dl_Ul(D, smesh, tmesh, source_landmarks,
                           target_landmarks, centroid, scale)

    # Parameters of the algorithm (Eq. 6)
    # order : Alpha, Beta, normal weighting, and max iteration for step
    if steps is None:
        steps = [
            [0.01, 10, 0.5, 10],
            [0.02, 5, 0.5, 10],
            [0.03, 2.5, 0.5, 10],
            [0.01, 0, 0.0, 10],
        ]
    if return_records:
        _, distances, _ = closest_point(tmesh, transformed_vertices)
        records = [(scale * transformed_vertices + centroid[None, :], distances)]

    # Main loop
    for i, (ws, wl, wn, max_iter) in enumerate(steps):

        # If normals are estimated from points and if there are less
        # than 3 points per query, avoid normal estimation
        if not use_faces and nb_neighbors < 3:
            wn = 0

        print(f'Step {i+1}/{len(steps)} : ws={ws}, wl={wl}, wn={wn}')
        last_error = np.finfo(np.float32).max
        error = np.finfo(np.float16).max
        cpt_iter = 0

        # Current step iterations loop
        while last_error - error > eps and (max_iter < 0 or cpt_iter < max_iter):

            if use_faces:
                # If using faces, use proximity.closest_point
                nearest, distances, tids = closest_point(tmesh, transformed_vertices)
                if wn > 0:
                    if vertex_normals:
                        # Normal interpolation
                        barys = points_to_barycentric(
                            tmesh.vertices[tmesh.faces[tids]], nearest)
                        tnormals = np.einsum('ij,ijk->ik',
                                             barys,
                                             tmesh.vertex_normals[tmesh.faces[tids]])
                    else:
                        # Face normal
                        tnormals = tmesh.face_normals[tids]
            else:
                # If using points, use scipy's cKDTree
                distances, indices = kdtree.query(transformed_vertices,
                                                  k=nb_neighbors,
                                                  workers=-1)
                nearest = tmesh.vertices[indices, :]
                if wn > 0:
                    # Estimate normal only if there is normal weighting
                    tnormals = plane_fit(nearest)[1]
                if nb_neighbors > 1:
                    # Focus on the closest match only per source vertex
                    nearest = nearest[:, 0]
                    distances = distances[:, 0]

            # Data weighting
            wd_vec = np.ones((len(transformed_vertices), 1))
            wd_vec[distances > distance_treshold] = 0

            if wn > 0:
                # Normal weighting = multiplying wd by cosines^wn
                snormals = DN * X
                dot = np.einsum('ij,ij->i', snormals, tnormals)
                # Normal orientation is only known for meshes
                if use_faces:
                    dot = np.clip(dot, 0, 1)
                else:
                    dot = np.abs(dot)
                wd_vec = wd_vec * dot[:, None] ** wn

            # Actual system solve
            X = _solve_system(M_kron_G, D, wd_vec, nearest, ws, nE, nV, Dl, Ul, wl)
            transformed_vertices = D * X
            last_error = error
            error_vec = np.linalg.norm(nearest - transformed_vertices, axis=-1)
            error = (error_vec * wd_vec).mean()
            if return_records:
                records.append((scale * transformed_vertices + centroid[None, :],
                                error_vec))
            cpt_iter += 1

    # Re-scale the meshes
    smesh.vertices = scale * smesh.vertices + centroid[None, :]
    tmesh.vertices = scale * tmesh.vertices + centroid[None, :]

    result = smesh.copy()
    result.vertices = scale * transformed_vertices + centroid[None, :]
    if return_records:
        return result, records
    return result
