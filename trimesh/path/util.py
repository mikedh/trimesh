import numpy as np


def is_ccw(points):
    """
    Check if connected planar points are counterclockwise.

    Parameters
    -----------
    points: (n,2) float, connected points on a plane

    Returns
    ----------
    ccw: bool, True if points are counterclockwise
    """
    points = np.asanyarray(points, dtype=np.float64)

    if (len(points.shape) != 2 or
            points.shape[1] != 2):
        raise ValueError('CCW is only defined for 2D')
    xd = np.diff(points[:, 0])
    yd = np.column_stack((
        points[:, 1],
        points[:, 1])).reshape(-1)[1:-1].reshape((-1, 2)).sum(axis=1)
    area = np.sum(xd * yd) * .5
    ccw = area < 0

    return ccw


def concatenate(paths):
    """
    Concatenate multiple paths into a single path.

    Parameters
    -------------
    paths: list of Path, Path2D, or Path3D objects

    Returns
    -------------
    concat: Path, Path2D, or Path3D object
    """
    # if only one path object just return copy
    if len(paths) == 1:
        return paths[0].copy()

    # length of vertex arrays
    vert_len = np.array([len(i.vertices) for i in paths])
    # how much to offset each paths vertex indices by
    offsets = np.append(0.0, np.cumsum(vert_len))[:-1].astype(np.int64)

    # resulting entities
    entities = []
    # resulting vertices
    vertices = []
    # resulting metadata
    metadata = {}
    for path, offset in zip(paths, offsets):
        # update metadata
        metadata.update(path.metadata)
        # copy vertices, we will stack later
        vertices.append(path.vertices.copy())
        # copy entity then reindex points
        for entity in path.entities:
            entities.append(entity.copy())
            entities[-1].points += offset

    # generate the single new concatenated path
    # use input types so we don't have circular imports
    concat = type(path)(metadata=metadata,
                        entities=entities,
                        vertices=np.vstack(vertices))
    return concat
