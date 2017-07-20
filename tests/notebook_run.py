import sys
import os
import json
import numpy as np


def load_notebook(file_obj,
                  exclude=['%', 'show', 'plt']):
    '''
    Load an ipynb file into a cleaned and stripped string that can be exec'd

    The motivation for this is to check ipynb examples with CI so they don't 
    get silently broken and confusing.

    Arguments
    ----------
    file_obj : open file object
    exclude  :  list, strs which if a line contains the line will be replaced
                by a pass statement. 

    Returns
    ----------
    script : str, cleaned script which can be passed to exec
    '''
    raw = json.load(file_obj)
    lines = []
    for line in np.hstack([i['source'] for i in raw['cells'] if 'source' in i]):
        if any(i in line for i in exclude):
            lines.append(to_pass(line))
        else:
            lines.append(line.rstrip())
    script = '\n'.join(lines) + '\n'
    return script


def to_pass(line):
    '''
    Replace a line of code with a pass statement, with
    the correct number of leading spaces

    Arguments
    ----------
    line : str, line of code

    Returns
    ----------
    passed : str, line of code with same leading spaces, but pass statement
    '''
    spaces = np.nonzero([i != ' ' for i in line])[0][0]
    passed = (' ' * spaces) + 'pass'
    return passed


if __name__ == '__main__':
    '''
    Load and run a notebook if a file name is passed
    '''
    file_name = sys.argv[-1]
    if (file_name.endswith('ipynb') and
        os.path.exists(file_name)):
        with open(file_name, 'r') as file_obj:
            script = load_notebook(file_obj)
        print('\nloaded {}:\n'.format(file_name))
        print(script)
        exec(script)
