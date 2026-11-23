import io

from ..exceptions import ExceptionWrapper
from ..typed import Any, Iterable, Number, Stream

# used as an intermediate format
from .gltf import load_glb

_cascade_step_formats = ("stp", "step")
_cascade_iges_formats = ("igs", "iges")
_cascade_zstd_formats = (
    "stp.zst",
    "stp.zstd",
    "step.zst",
    "step.zstd",
    "igs.zst",
    "igs.zstd",
    "iges.zst",
    "iges.zstd",
)
_cascade_formats = _cascade_step_formats + _cascade_iges_formats + _cascade_zstd_formats


def load_step(
    file_obj: Stream,
    file_type: str | None = None,
    tol_linear: Number | None = None,
    tol_angular: Number | None = None,
    tol_relative: bool | None = None,
    merge_primitives: bool = True,
    use_parallel: bool | None = None,
    include_brep: bool | None = None,
    brep_types: Iterable[str] | None = None,
    include_materials: bool | None = None,
    node_name_format: Any = None,
    y_up: bool | None = None,
    **kwargs,
) -> dict:
    """
    Use `cascadio`, a packaged version of OpenCASCADE, to load a STEP or IGES
    file completely in memory using GLB as an intermediate format.

    Parameters
    ----------
    file_obj
      STEP or IGES file-like object to load (plain or Zstandard-compressed).
    file_type
      File format extension (e.g. ``"step"``, ``"stp"``, ``"iges"``, ``"igs"``,
      or their ``.zst`` variants).
    tol_linear
      Maximum linear deflection between the mesh and true surface.
    tol_angular
      Maximum angular deflection between adjacent mesh segments in radians.
    tol_relative
      If True, ``tol_linear`` is relative to edge length.
    merge_primitives
      Produce one mesh primitive per part instead of one per face.
    use_parallel
      Use multiple threads for meshing and export.
    include_brep
      Embed the ``TM_brep_faces`` extension (per-triangle CAD face index,
      analytic surface geometry, and part substance). Implies ``merge_primitives=True``.
    brep_types
      Subset of surface types to include when ``include_brep=True``
      (any of ``"plane"``, ``"cylinder"``, ``"cone"``, ``"sphere"``, ``"torus"``).
    include_materials
      In ``cascadio >= 0.2``, equivalent to ``include_brep``.
    node_name_format
      ``cascadio.NodeNameFormat`` value controlling GLB node names (``cascadio >= 0.2``).
    y_up
      Rotate output from CAD Z-up into glTF Y-up (``cascadio >= 0.2``).
    **kwargs
      ``timeout`` in seconds and ``memory_limit`` in bytes are passed to
      ``cascadio >= 0.2`` when given, ``None`` disabling either, and the
      rest to ``trimesh.exchange.gltf.load_glb``.

    Returns
    -------
    kwargs
      Keyword arguments for a ``Scene``.
    """
    data = file_obj.read()
    if isinstance(data, str):
        data = data.encode("utf-8")

    raw_type = (file_type or "step").lower()
    kind = raw_type.removesuffix(".zstd").removesuffix(".zst")

    # build kwargs explicitly so defaults stay owned by cascadio and
    # 0.2-only options are omitted unless the caller passed them
    cascadio_kwargs: dict[str, Any] = {
        "file_type": kind,
        "merge_primitives": bool(merge_primitives),
    }
    optional = {
        "tol_linear": tol_linear,
        "tol_angular": tol_angular,
        "tol_relative": tol_relative,
        "use_parallel": use_parallel,
        "include_brep": include_brep,
        "include_materials": include_materials,
        "node_name_format": node_name_format,
        "y_up": y_up,
    }
    cascadio_kwargs.update(
        {key: value for key, value in optional.items() if value is not None}
    )
    if brep_types is not None:
        cascadio_kwargs["brep_types"] = set(brep_types)
    # limits only when asked for, as an explicit `None` disables them
    cascadio_kwargs.update(
        {key: kwargs.pop(key) for key in ("timeout", "memory_limit") if key in kwargs}
    )

    if hasattr(cascadio, "load"):
        glb = cascadio.load(data, **cascadio_kwargs)
    else:
        # cascadio<0.2 fallback: delete with its deprecation
        import os
        import tempfile

        legacy = (
            "tol_linear",
            "tol_angular",
            "tol_relative",
            "merge_primitives",
            "use_parallel",
        )
        with tempfile.TemporaryDirectory() as directory:
            step_path = os.path.join(directory, "data.step")
            glb_path = os.path.join(directory, "converted.glb")
            with open(step_path, "wb") as f:
                f.write(data)
            cascadio.step_to_glb(
                step_path,
                glb_path,
                **{key: cascadio_kwargs[key] for key in legacy if key in cascadio_kwargs},
            )
            with open(glb_path, "rb") as f:
                glb = f.read()

    if len(glb) == 0:
        # cascadio 0.1.x returned empty bytes on error instead of raising
        raise ValueError("`cascadio` returned empty GLB data")

    return load_glb(
        file_obj=io.BytesIO(glb),
        merge_primitives=merge_primitives,
        **kwargs,
    )


cascadio: Any = None
try:
    # wheels for most platforms: `pip install cascadio`
    import cascadio

    _supported: tuple[str, ...] = _cascade_step_formats
    if hasattr(cascadio, "load"):
        _supported += _cascade_iges_formats
    if hasattr(cascadio, "DEFAULT_TIMEOUT"):
        _supported += _cascade_zstd_formats
    _cascade_loaders = dict.fromkeys(_supported, load_step)
except BaseException as E:
    wrapper = ExceptionWrapper(E)
    _cascade_loaders = dict.fromkeys(_cascade_formats, wrapper)
