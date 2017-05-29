import numpy as np

from trimesh import util

# a matrix where all non- diagonal terms are -1.0
# and all diagonal terms are 1.0
negate_nondiagonal = (np.eye(3, dtype=np.float64) * 2) - 1


def cylinder_inertia(mass, radius, height, transform=None):
    '''
    Return the inertia tensor of a cylinder.

    Parameters
    ------------
    mass:      float, mass of cylinder
    radius:    float, radius of cylinder
    height:    float, height of cylinder
    transform: (4,4) float, transformation of cylinder 

    Returns
    ------------
    inertia: (3,3) float, inertia tensor
    '''
    h2, r2 = height ** 2, radius ** 2
    diagonal = np.array([((mass * h2) / 12) + ((mass * r2) / 4),
                         ((mass * h2) / 12) + ((mass * r2) / 4),
                         (mass * r2) / 2])
    inertia = diagonal * np.eye(3)

    if transform is not None:
        inertia = transform_inertia(transform, inertia)

    return inertia


def sphere_inertia(mass, radius):
    '''
    Return the inertia tensor of a sphere.

    Parameters
    ------------
    mass:      float, mass of sphere
    radius:    float, radius of sphere

    Returns
    ------------
    inertia: (3,3) float, inertia tensor
    '''
    inertia = (2.0 / 5.0) * (radius ** 2) * mass * np.eye(3)
    return inertia


def principal_axis(inertia):
    '''
    Find the principal components and principal axis 
    of inertia from the inertia tensor.

    Parameters
    ------------
    inertia: (3,3) float, inertia tensor

    Returns
    ------------
    components: (3,) float, principal components of inertia
    vectors:    (3,3) float, row vectors pointing along 
                             the principal axes of inertia
    '''
    inertia = np.asanyarray(inertia, dtype=np.float64)
    if inertia.shape != (3, 3):
        raise ValueError('inertia tensor must be (3,3)!')

    components, vectors = np.linalg.eig(inertia * negate_nondiagonal)

    # eig returns them as column vectors, change them to row vectors
    vectors = vectors.T

    return components, vectors


def transform_inertia(transform, inertia_tensor):
    '''
    Transform an inertia tensor to a new frame. 

    More details in OCW PDF: 
    MIT16_07F09_Lec26.pdf

    Parameters
    ------------
    transform:      (3,3) or (4,4) float, transformation matrix
    inertia_tensor: (3,3) float, inertia tensor

    Returns
    ------------
    transformed: (3,3) float, inertia tensor in new frame
    '''
    transform = np.asanyarray(transform, dtype=np.float64)
    if transform.shape == (4, 4):
        rotation = transform[:3, :3]
    elif transform.shape == (3, 3):
        rotation = transform
    else:
        raise ValueError('transform must be (3,3) or (4,4)!')

    inertia_tensor = np.asanyarray(inertia_tensor, dtype=np.float64)
    if inertia_tensor.shape != (3, 3):
        raise ValueError('inertia_tensor must be (3,3)!')

    transformed = util.multi_dot([rotation,
                                  inertia_tensor * negate_nondiagonal,
                                  rotation.T])
    transformed *= negate_nondiagonal
    return transformed
