from .import ray_triangle

# add to __all__ as per pep8
__all__ = [ray_triangle]

# optionally load an interface to the embree raytracer
try:
    from . import ray_pyembree
    has_embree = True
    __all__.append(ray_pyembree)
except ImportError:
    has_embree = False
