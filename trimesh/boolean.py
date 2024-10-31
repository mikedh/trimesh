"""
boolean.py
-------------

Do boolean operations on meshes using either Blender or Manifold.
"""

# for dynamic module imports
from importlib import import_module
from .typed import Optional, Sequence


def difference(
    meshes: Sequence, engine: Optional[str] = None, check_volume: bool = True, **kwargs
):
    """
    Compute the boolean difference between a mesh an n other meshes.

    Parameters
    ----------
    meshes : sequence of trimesh.Trimesh
      Meshes to be processed.
    engine
      Which backend to use, i.e. 'blender' or 'manifold'
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.
    kwargs
      Passed through to the `engine`.

    Returns
    ----------
    difference
      A `Trimesh` that contains `meshes[0] - meshes[1:]`
    """
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")

    return _engines[engine](meshes, operation="difference", **kwargs)


def union(
    meshes: Sequence, engine: Optional[str] = None, check_volume: bool = True, **kwargs
):
    """
    Compute the boolean union between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'manifold'
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.
    kwargs
      Passed through to the `engine`.

    Returns
    ----------
    union
      A `Trimesh` that contains the union of all passed meshes.
    """
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")

    return _engines[engine](meshes, operation="union", **kwargs)


def intersection(
    meshes: Sequence, engine: Optional[str] = None, check_volume: bool = True, **kwargs
):
    """
    Compute the boolean intersection between a mesh and other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'manifold'
    check_volume
      Raise an error if not all meshes are watertight
      positive volumes. Advanced users may want to ignore
      this check as it is expensive.
    kwargs
      Passed through to the `engine`.

    Returns
    ----------
    intersection
      A `Trimesh` that contains the intersection geometry.
    """
    if check_volume and not all(m.is_volume for m in meshes):
        raise ValueError("Not all meshes are volumes!")
    return _engines[engine](meshes, operation="intersection", **kwargs)


# supported backend boolean engines
# ordered by preference
all_engines = [
    "manifold",
    "blender",
]

# test which engines are actually usable
_engines = {}
available_engines = []
for name in all_engines:
    engine = import_module("trimesh.interfaces." + name)
    if engine.exists:
        _engines[name] = engine.boolean
        available_engines.append(name)
        if None not in _engines:
            # pick the first available engine as the default
            _engines[None] = engine.boolean

def set_default_engine(engine):
    if engine not in available_engines:
        raise ValueError(f"Engine {engine} is not available on this system")
    _engines[None] = _engines[engine]
