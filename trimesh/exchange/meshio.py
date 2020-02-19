def load_meshio(file_obj, file_type=None, **kwargs):
    """
    Load a meshio-supported file into the kwargs for a Trimesh constructor


    Parameters
    ----------
    file_obj : file object
      Contains a meshio file
    file_type : str
      File extension, aka 'vtk'

    Returns
    ----------
    loaded : dict
      kwargs for Trimesh constructor
    """
    # trimesh "file types" are really filename extensions
    file_type = meshio.extension_to_filetype["." + file_type]
    # load_meshio gets passed and io.BufferedReader, but not all readers can cope with
    # that, e.g., the ones that use h5m underneath. Use the associated file name
    # instead.
    mesh = meshio.read(file_obj.name, file_format=file_type)

    # save data as kwargs for a trimesh.Trimesh
    result = {'vertices': mesh.points,
              'faces': mesh.get_cells_type("triangle")}
    return result


try:
    import meshio
except BaseException:
    _meshio_loaders = {}
else:
    _meshio_formats = [ext[1:] for ext in set(meshio.extension_to_filetype.keys())]
    _meshio_loaders = dict(zip(_meshio_formats,
                               [load_meshio] * len(_meshio_formats)))
