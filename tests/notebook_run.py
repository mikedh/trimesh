import sys
import os
import json
import inspect
import subprocess
import numpy as np

# current working directory
cwd = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))


def load_notebook(file_obj,
                  exclude=['%', 'show', 'plt']):
    """
    Load an ipynb file into a cleaned and stripped string that can
    be ran with `exec`

    The motivation for this is to check ipynb examples with CI so
    they don't get silently broken and confusing.

    Arguments
    ----------
    file_obj :  open file object
    exclude  :  list, strs which if a line contains the line
                will be replaced by a pass statement.

    Returns
    ----------
    script : str, cleaned script which can be passed to exec
    """
    raw = json.load(file_obj)
    lines = []
    for line in np.hstack([i['source']
                           for i in raw['cells'] if 'source' in i]):
        if any(i in line for i in exclude):
            lines.append(to_pass(line))
        else:
            lines.append(line.rstrip())
    script = '\n'.join(lines) + '\n'
    return script


def to_pass(line):
    """
    Replace a line of code with a pass statement, with
    the correct number of leading spaces

    Arguments
    ----------
    line : str, line of code

    Returns
    ----------
    passed : str, line of code with same leading spaces
                  but code replaced with pass statement
    """
    spaces = np.nonzero([i != ' ' for i in line])[0][0]
    passed = (' ' * spaces) + 'pass'
    return passed


def render_notebook(file_name, out_name, nbconvert='jupyter'):
    """
    Render an IPython notebook to an HTML file.
    """
    out_name = os.path.abspath(out_name)
    file_name = os.path.abspath(file_name)

    command = [nbconvert,
               'nbconvert',
               '--to',
               'html',
               file_name,
               '--output',
               out_name]

    subprocess.check_call(command)


def render_examples(out_dir, in_dir=None, ext='ipynb'):
    """
    Render all IPython notebooks in a directory to HTML.
    """
    # replace with relative path
    if in_dir is None:
        in_dir = os.path.abspath(
            os.path.join(cwd, '../examples'))

    for file_name in os.listdir(in_dir):
        # check extension
        split = file_name.split('.')
        if split[-1] != ext:
            continue
        # full path of file
        nb_path = os.path.join(in_dir, file_name)

        html_path = os.path.join(out_dir,
                                 '.'.join(split[:-1]) + '.html')
        render_notebook(nb_path, html_path)


if __name__ == '__main__':
    """
    Load and run a notebook if a file name is passed
    """
    if "examples" in sys.argv:
        out_path = sys.argv[sys.argv.index("examples") + 1]
        render_examples(out_path)
    elif "exec" in sys.argv:
        # exec the script passed
        file_name = sys.argv[sys.argv.index("exec") + 1]
        if (file_name.endswith('ipynb') and
                os.path.exists(file_name)):
            with open(file_name, 'r') as file_obj:
                script = load_notebook(file_obj)
            print('\nloaded {}:\n'.format(file_name))
            print(script)
            exec(script)
