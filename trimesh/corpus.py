"""
corpus.py
----------

Deal with large remote batches of meshes.
"""

from .resolvers import Resolver


class Corpus(Resolver):
    """
    A corpus is a superset of a file object resolver
    which represents a collection
    """

    def __init__(self, cache=True, cache_path=None):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
