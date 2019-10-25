import numpy as np


def load_xyz(file_obj,
             delimiter=' ',
             *args,
             **kwargs):
    """
    Load a XYZ file from an open file object.
    
    Parameters
    ----------
    file_obj : an open file-like object 
      Source data, ASCII XYZ
    separator : string
      Symol(s) used to separate the columns of the file

    Returns
    -------

    pointcloud_kwargs : dict
      Data which can be passed to 
      Pointcloud constructor, eg: c = Pointcloud(**pointcloud_kwargs)
    """
    data = np.loadtxt(file_obj, ndmin=2, delimiter=delimiter)
    num_cols = len(data[0])
    vertices = data[:, :3]
    if num_cols == 3:
        # only positions
        colors = None
    elif num_cols == 4:
        # color given by scalar value, map to color?
        colors = None
    elif num_cols == 6:
        colors = np.array(data[:, 3:], dtype=np.uint8)
        colors = np.concatenate((colors,
                                 np.ones((len(data), 1), dtype=np.uint8)*255),
                                axis=1)
    elif num_cols == 7:
        colors = np.array(data[:, 3:], dtype=np.uint8)
    else:
        raise ValueError("Unknown data type in xyz file")
    result = {'vertices': vertices,
              'colors': colors,
              'metadata': {}}
    return result

def export_xyz(cloud, write_colors=True, delimiter=' '):
    data = cloud.vertices
    num_cols = 3
    
    if write_colors and cloud.colors is not None:
        data = np.concatenate((data, cloud.colors), axis=1)
        num_cols += 4

    fmt = (('{}' + delimiter)*num_cols)[:-1]
    export = ((fmt+'\n')*len(data))[:-1].format(*data.flatten())
    return export
               

_xyz_loaders = {'xyz': load_xyz}
_xyz_exporters = {'xyz': export_xyz}
