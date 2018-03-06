from .import ray_triangle

# optionally load an interface to the embree raytracer
try:
    from . import ray_pyembree
    has_embree = True
except ImportError:
    has_embree = False
