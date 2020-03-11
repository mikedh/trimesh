"""
Build trimesh docs using subprocess to call sphinx.

To build docs:
```
python build.py
```
"""

import os
import sys
import shutil
import inspect
import subprocess


def abspath(rel):
    """
    Take paths relative to the current file and
    convert them to absolute paths.

    Parameters
    ------------
    rel     : str
              Relative path, IE '../stuff'

    Returns
    -------------
    abspath : str
              Absolute path, IE '/home/user/stuff'
    """
    return os.path.abspath(os.path.join(cwd, rel))


# current working directory
cwd = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

# output location for all the html
build_dir = abspath('html')

# sphinx produces a lot of output
verbose = '-v' in sys.argv

if __name__ == '__main__':
    # convert README to an RST for sphinx
    pdc = ['pandoc',
           '--from=markdown',
           '--to=rst',
           '--output=' + abspath('README.rst'),
           abspath('../README.md')]
    subprocess.check_call(pdc)

    # render IPython examples to HTML
    # create examples directory
    examples_dir = os.path.join(build_dir, 'examples')
    try:
        os.makedirs(examples_dir)
    except FileExistsError:
        pass
    exp = ['python',
           abspath('../tests/notebooks.py'),
           'examples',
           examples_dir]
    subprocess.check_call(exp)

    # copy images to build director
    try:
        shutil.copytree(
            abspath('images'),
            os.path.join(build_dir, 'images'))
    except BaseException as E:
        print('unable to copy images', E)

    # build the API doc
    api = ['sphinx-apidoc',
           '-e',
           '-o',
           cwd,
           abspath('../trimesh')]

    # build the HTML docs
    bld = ['sphinx-build',
           '-b',
           'html',
           cwd,
           build_dir]

    # sphinx produces a ton of useless output
    subprocess.check_call(api)
    subprocess.check_call(bld)

    # keep github pages from using jekyll
    with open(os.path.join(build_dir, '.nojekyll'), 'w') as f:
        f.write(' ')

    # add cname for docs domain
    with open(os.path.join(build_dir, 'CNAME'), 'w') as f:
        f.write('trimsh.org')
