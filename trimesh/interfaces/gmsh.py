import os
import tempfile
import warnings

import numpy as np

from ..constants import log
from ..typed import Integer, Number, Optional

# the old `gmsh-sdk` package is deprecated and
# has a different incompatible API!
_min_gmsh = (4, 12, 1)


_warning = " ".join(
    [
        "`trimesh.interfaces.gmsh` is deprecated and will be removed January 2025!",
        "There are *many* gmsh options on PyPi: `scikit-gmsh` `gmsh` `pygmsh` `gmsh-sdk`,",
        "users should pick one of those and use it directly. If STEP loading is the only",
        "thing needed you may want `pip install cascadio` which uses OpenCASCADE more",
        "directly and will immediately enable STEP as a loadable format in trimesh.",
    ]
)


def load_gmsh(file_name, gmsh_args=None, interruptible=True):
    """
    Returns a surface mesh from CAD model in Open Cascade
    Breap (.brep), Step (.stp or .step) and Iges formats
    Or returns a surface mesh from 3D volume mesh using gmsh.

    For a list of possible options to pass to GMSH, check:
    http://gmsh.info/doc/texinfo/gmsh.html

    An easy way to install the GMSH SDK is through the `gmsh`
    package on PyPi, which downloads and sets up gmsh:
        >>> pip install gmsh

    Parameters
    --------------
    file_name : str
      Location of the file to be imported
    gmsh_args : (n, 2) list
      List of (parameter, value) pairs to be passed to
      gmsh.option.setNumber
    max_element : float or None
      Maximum length of an element in the volume mesh
    interruptible : bool
      Allows load_gmsh to run outside of the main thread if False,
      default behaviour if set to True. Added in 4.12.0

    Returns
    ------------
    mesh : trimesh.Trimesh
      Surface mesh of input geometry
    """
    warnings.warn(_warning, category=DeprecationWarning, stacklevel=2)

    # use STL as an intermediate format
    # do import here to avoid very occasional segfaults
    import gmsh

    # the deprecated `pip install gmsh-sdk` package has an entirely different API
    # require a minimum version here
    if tuple(int(i) for i in gmsh.__version__.split(".")) < _min_gmsh:
        raise ImportError(
            f"`gmsh.__version__ < {_min_gmsh}`: run `pip install --upgrade gmsh`"
        )

    from ..exchange.stl import load_stl

    # start with default args for the meshing step
    # Mesh.Algorithm=2 MeshAdapt/Delaunay, there are others but they may include quads
    # With this planes are meshed using Delaunay and cylinders are meshed
    # using MeshAdapt
    args = [
        ("Mesh.Algorithm", 2),
        ("Mesh.CharacteristicLengthFromCurvature", 1),
        ("Mesh.MinimumCirclePoints", 32),
    ]
    # add passed argument tuples last so we can override defaults
    if gmsh_args is not None:
        args.extend(gmsh_args)

    # formats GMSH can load
    supported = [
        ".brep",
        ".stp",
        ".step",
        ".igs",
        ".iges",
        ".bdf",
        ".msh",
        ".inp",
        ".diff",
        ".mesh",
    ]

    # check extensions to make sure it is supported format
    if file_name is not None:
        if not any(file_name.lower().endswith(e) for e in supported):
            raise ValueError(
                "Supported formats are: BREP (.brep), STEP (.stp or .step), "
                + "IGES (.igs or .iges), Nastran (.bdf), Gmsh (.msh), Abaqus (*.inp), "
                + "Diffpack (*.diff), Inria Medit (*.mesh)"
            )
    else:
        raise ValueError("No import since no file was provided!")

    # hmmm
    init = False
    try:
        if hasattr(gmsh, "isInitialized"):
            init = gmsh.isInitialized()
        elif hasattr(gmsh, "is_initialized"):
            init = gmsh.is_initialized()
    except BaseException:
        log.debug("gmsh unexpected", exc_info=True)

    if not init:
        gmsh.initialize(interruptible=interruptible)

    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Surface_Mesh_Generation")
    # loop through our numbered args which do things, stuff
    for arg in args:
        gmsh.option.setNumber(*arg)

    gmsh.open(file_name)

    # create a temporary file for the results
    out_data = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    # windows gets mad if two processes try to open the same file
    out_data.close()

    # we have to mesh the surface as these are analytic BREP formats
    if any(
        file_name.lower().endswith(e) for e in [".brep", ".stp", ".step", ".igs", ".iges"]
    ):
        gmsh.model.geo.synchronize()
        # generate the mesh
        gmsh.model.mesh.generate(2)
        # write to the temporary file
        gmsh.write(out_data.name)
    else:
        gmsh.plugin.run("NewView")
        gmsh.plugin.run("Skin")
        gmsh.view.write(1, out_data.name)

    # load the data from the temporary outfile
    with open(out_data.name, "rb") as f:
        kwargs = load_stl(f)

    gmsh.finalize()

    return kwargs


def to_volume(
    mesh,
    file_name: Optional[str] = None,
    file_type: Optional[str] = None,
    max_element: Optional[Number] = None,
    mesher_id: Integer = 1,
) -> bytes:
    """
    Convert a surface mesh to a 3D volume mesh generated by gmsh.

    An easy way to install the gmsh sdk is through the gmsh
    package on pypi, which downloads and sets up gmsh:
        pip install gmsh

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
    file_type
      Location to save output, in .msh (gmsh) or .bdf (Nastran) format
    max_element : float or None
      Maximum length of an element in the volume mesh
    mesher_id : int
      3D unstructured algorithms:
      1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT

    Returns
    ------------
    data : None or bytes
      MSH data, only returned if file_name is None

    """
    warnings.warn(_warning, category=DeprecationWarning, stacklevel=2)

    # do import here to avoid very occasional segfaults
    import gmsh

    if tuple(int(i) for i in gmsh.__version__.split(".")) < _min_gmsh:
        raise ImportError(
            f"`gmsh.__version__ < {_min_gmsh}`: run `pip install --upgrade gmsh`"
        )

    # checks mesher selection
    if mesher_id not in [1, 3, 4, 7, 9, 10]:
        raise ValueError("unavailable mesher selected!")
    else:
        mesher_id = int(mesher_id)

    # set max element length to a best guess if not specified
    if max_element is None:
        max_element = np.sqrt(np.mean(mesh.area_faces))

    if file_name is not None:
        file_type = file_name.split(".")[-1].lower()
    elif file_type is not None:
        file_type = file_type.lower().strip(".")
    else:
        file_type = "msh"

    if file_type not in ["bdf", "msh", "inp", "diff", "mesh"]:
        raise ValueError(
            "Only Nastran (`bdf`), Gmsh (`msh`), Abaqus (`inp`), "
            + "Diffpack (`diff`) and Inria Medit (`mesh`) formats "
            + "are available!"
        )

    # use a temporary directory for input and output
    with tempfile.TemporaryDirectory() as D:
        mesh_file = os.path.join(D, "input.stl")
        mesh.export(mesh_file)

        # starts Gmsh Python API script
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("Nastran_stl")

        gmsh.merge(mesh_file)
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
        gmsh.model.setPhysicalName(dim, p2, "Nastran_bdf")

        out_path = os.path.join(D, f"temp.{file_type}")
        gmsh.write(out_path)
        with open(out_path, "rb") as f:
            data = f.read()

    # close up shop
    gmsh.finalize()

    # write the data
    if file_name is not None:
        with open(os.path.abspath(os.path.expanduser(file_name)), "wb") as f:
            f.write(data)

    return data
