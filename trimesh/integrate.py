"""
integrate.py
--------------

Utilities for integrating functions over meshes surfaces.
"""

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr as sympy_parse

from .constants import log
from . import util

import warnings
warnings.warn(
    '`trimesh.integrate` is deprecated and will ' +
    'be removed in March 2024, you might take a ' +
    'look at `quadpy` which is far more complete',
    category=DeprecationWarning,
    stacklevel=2)


def symbolic_barycentric(function):
    """
    Symbolically integrate a function(x,y,z) across a triangle or mesh.

    Parameters
    ----------
    function: string or sympy expression
              x, y, z will be replaced with a barycentric representation
              and the the function is integrated across the triangle.

    Returns
    ----------
    evaluator: numpy lambda function of result which takes a mesh
    expr:      sympy expression of result

    Examples
    -----------

    In [1]: function = '1'

    In [2]: integrator, expr = integrate_barycentric(function)

    In [3]: integrator
    Out[3]: <__main__.evaluator instance at 0x7f66cd2a6200>

    In [4]: expr
    Out[4]: 1/2

    In [5]: result  = integrator(mesh)

    In [6]: mesh.area
    Out[6]: 34.641016151377542

    In [7]: result.sum()
    Out[7]: 34.641016151377542
    """

    class evaluator:

        def __init__(self, expr, expr_args):
            self.lambdified = sp.lambdify(args=expr_args,
                                          expr=expr,
                                          modules=['numpy', 'sympy'])

        def __call__(self, mesh, *args):
            """
            Quickly evaluate the surface integral across a mesh

            Parameters
            ----------
            mesh: Trimesh object

            Returns
            ----------
            integrated: (len(faces),) float, integral evaluated for each face
            """
            integrated = self.lambdified(*mesh.triangles.reshape((-1, 9)).T)
            integrated *= 2 * mesh.area_faces
            return integrated

    function, symbols = substitute_barycentric(function)

    b1, b2, x1, x2, x3, y1, y2, y3, z1, z2, z3 = symbols
    # do the first integral for b1
    integrated_1 = sp.integrate(function, b1)
    integrated_1 = (integrated_1.subs({b1: 1 - b2}) -
                    integrated_1.subs({b1: 0}))

    integrated_2 = sp.integrate(integrated_1, b2)
    integrated_2 = (integrated_2.subs({b2: 1}) -
                    integrated_2.subs({b2: 0}))

    lambdified = evaluator(expr=integrated_2,
                           expr_args=[x1, y1, z1, x2, y2, z2, x3, y3, z3])

    return lambdified, integrated_2


def substitute_barycentric(function):
    if util.is_string(function):
        function = sympy_parse(function)
    # barycentric coordinates
    b1, b2 = sp.symbols('b1 b2',
                        real=True,
                        positive=True)
    # vertices of the triangles
    x1, x2, x3, y1, y2, y3, z1, z2, z3 = sp.symbols(
        'x1,x2,x3,y1,y2,y3,z1,z2,z3', real=True)

    # generate the substitution dictionary to convert from cartesian to barycentric
    # since the input could have been a sympy expression or a string
    # that we parsed substitute based on name to avoid id(x) issues
    substitutions = {}
    for symbol in function.free_symbols:
        if symbol.name == 'x':
            substitutions[symbol] = b1 * x1 + b2 * x2 + (1 - b1 - b2) * x3
        elif symbol.name == 'y':
            substitutions[symbol] = b1 * y1 + b2 * y2 + (1 - b1 - b2) * y3
        elif symbol.name == 'z':
            substitutions[symbol] = b1 * z1 + b2 * z2 + (1 - b1 - b2) * z3
    # apply the conversion to barycentric
    function = function.subs(substitutions)
    log.debug('converted function to barycentric: %s', str(function))

    symbols = (b1, b2, x1, x2, x3, y1, y2, y3, z1, z2, z3)

    return function, symbols
