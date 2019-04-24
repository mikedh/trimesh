"""
notebook.py
-------------

Render trimesh.Scene objects in HTML
and jupyter notebooks using three.js
"""
import os
import base64

# for our template
from ..resources import get_resource


def scene_to_html(scene):
    """
    Return HTML that will render the scene using
    GLTF/GLB encoded to base64 loaded by three.js

    Parameters
    --------------
    scene : trimesh.Scene
      Source geometry

    Returns
    --------------
    html : str
      HTML containing embedded geometry
    """
    # use os.path.join so this works on windows
    base = get_resource('viewer.html.template')

    # get export as bytes
    data = scene.export(file_type='glb')
    # encode as base64 string
    encoded = base64.b64encode(data).decode('utf-8')

    # replace keyword with our scene data
    result = base.replace('$B64GLTF', encoded)

    return result


def scene_to_notebook(scene, height=500, **kwargs):
    """
    Convert a scene to HTML containing embedded geometry
    and a three.js viewer that will display nicely in
    an IPython/Jupyter notebook.

    Parameters
    -------------
    scene : trimesh.Scene
      Source geometry

    Returns
    -------------
    html : IPython.display.HTML
      Object containing rendered scene
    """
    # keep as soft dependency
    from IPython import display

    # convert scene to a full HTML page
    as_html = scene_to_html(scene=scene)

    # escape the quotes in the HTML
    srcdoc = as_html.replace('"', '&quot;')
    # embed this puppy as the srcdoc attr of an IFframe
    # I tried this a dozen ways and this is the only one that works
    # display.IFrame/display.Javascript really, really don't work
    embedded = display.HTML(
        '<iframe srcdoc="{srcdoc}" '
        'width="100%" height="{height}px" '
        'style="border:none;"></iframe>'.format(
            srcdoc=srcdoc,
            height=height))
    return embedded


def in_notebook():
    """
    Check to see if we are in an IPython or Jypyter notebook.

    Returns
    -----------
    in_notebook : bool
      Returns True if we are in a notebook
    """
    try:
        # function returns IPython context, but only in IPython
        ipy = get_ipython()  # NOQA
        # we only want to render rich output in notebooks
        # in terminals we definitely do not want to output HTML
        name = str(ipy.__class__).lower()
        terminal = 'terminal' in name

        # spyder uses ZMQshell, and can appear to be a notebook
        spyder = '_' in os.environ and 'spyder' in os.environ['_']

        # assume we are in a notebook if we are not in
        # a terminal and we haven't been run by spyder
        notebook = (not terminal) and (not spyder)

        return notebook

    except BaseException:
        return False
