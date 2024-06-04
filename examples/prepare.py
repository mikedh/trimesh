"""
An example of a simple way of preparing and analyzing a model
for properties of interest to 3D printing, implemented using
dataclasses which can easily be used for API routes.

- Volume
- Oriented Bounding Box
- thin regions

The nominal operations flow we're using here is defined by
easily serialized `dataclass` objects and goes:

RawFile ->           # a raw loadable file
  PrintCandidate ->  # usable properties about the model
    PrintJob ->      # after quantities and options are picked
      PrintRelease   # i.e. what you would send to the machine
"""

import os
from dataclasses import dataclass
from typing import Optional

import trimesh


@dataclass
class RawFile:
    # the data for the loadable file
    file_data: bytes

    # the original name of the file
    # the extension will be used to determine type
    file_name: str

    # the units of the file if it is in a format
    # that does not include units (i.e. STL files)
    units: Optional[str] = None


@dataclass
class Error:
    # the user-facing message
    message: str

    # a code that can be used to group problems
    code: str

    # if faces were colored to indicate the problem
    color: str


@dataclass
class PrintCandidate:
    # the repaired model repaired into a GLB file
    glb: bytes

    # any problems that were encountered
    errors: list[Error]

    # the volume of the mesh
    volume: float

    # the size of the axis-aligned bounding box
    extents: list[float]


def load(raw: RawFile) -> trimesh.Trimesh:
    """
    Load a raw file for 3D printing, collapsing meshes into a single body.
    """
    # coerce scenes and formats with normals into a mesh
    mesh = trimesh.load(
        file_obj=trimesh.util.wrap_as_stream(raw.file_data),
        file_type=raw.file_name,
        process=True,
        merge_tex=True,  # ignore UV coordinates
        merge_norm=True,  # ignore vertex normals
        force="mesh",  # concatenate into single mesh
    )
    mesh.process(merge_tex=True, merge_norm=True)

    # if passed explicitly
    if mesh.units is None and raw.units is not None:
        mesh.units = raw.units

    # convert mesh to meters and guess as a fallback
    mesh.convert_units("meters", guess=True)

    # make sure it is using face colors
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    return mesh


def prepare(
    mesh: trimesh.Trimesh,
    minimum_thickness: float = 0.0005,
) -> PrintCandidate:
    """
    Take a raw 3D model and process it for 3D printing.
    """

    # collect issues to report
    problems = []

    if not mesh.fill_holes():
        problems.append(
            Error(message="Mesh is not watertight!", code="watertight", color="#FF0000")
        )
        # will apply "red" to broken faces
        trimesh.repair.broken_faces(mesh=mesh, color=[255, 0, 0, 255])

    thick_bad = (
        trimesh.proximity.thickness(
            mesh, mesh.triangles_center - mesh.face_normals * 0.001
        )
        < minimum_thickness
    )

    if thick_bad.any():
        mesh.visual.face_colors[thick_bad] = [0, 0, 255, 255]
        problems.append(
            Error(
                message="Mesh has regions below minimum thickness!",
                code="watertight",
                color="#0000FF",
            )
        )

    mesh.show()

    return PrintCandidate(
        glb=mesh.export(file_type="glb"),
        errors=problems,
        volume=mesh.volume,
        extents=mesh.extents,
    )


if __name__ == "__main__":
    # current working directory
    cwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))

    # create a raw "request"
    file_name = "machinist.3DXML"
    # file_name = 'ADIS16480.STL'
    file_path = os.path.join(cwd, "..", "models", file_name)
    with open(file_path, "rb") as f:
        loaded = load(raw=RawFile(file_data=f.read(), file_name=file_name, units=None))

    from pyinstrument import Profiler

    with Profiler() as P:
        report = prepare(loaded)

    P.print()
