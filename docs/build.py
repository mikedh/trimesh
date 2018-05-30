"""
Build trimesh docs using subprocess to call sphinx.

To build docs:
```
python build.py
```
"""

import os
import inspect
import subprocess

def abspath(rel):
    """
    Take paths relative to the current file
    and convert them to absolute paths.
    """
    return os.path.abspath(os.path.join(cwd, rel))

# current working directory
cwd = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

# output location for all the html
build_dir = abspath('html')


if __name__ == '__main__':

    # convert main readme to an RST for docs
    pdc = ['pandoc',
           '--from=markdown',
           '--to=rst',
           '--output=' + abspath('README.rst'),
           abspath('../README.md')]
    subprocess.check_call(pdc)
    
    # render IPython examples to HTML
    # create examples directory
    examples_dir = os.path.join(build_dir, 'examples')
    os.makedirs(examples_dir)
    exp = ['python',
           abspath('../tests/notebook_run.py'),
           'examples',
           examples_dir]    
    subprocess.check_call(exp)
    

    # build the API doc
    api = ['sphinx-apidoc',
           '-o',
           cwd,
           abspath('../trimesh')]
    subprocess.check_call(api)

    
    # build the HTML docs
    bld = ['sphinx-build',
           '-b',
           'html',
           cwd,
           build_dir]
    subprocess.check_call(bld)
