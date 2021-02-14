"""
dxf_helper.py
---------------

Manipulate DXF templates as plain text files rather
than strings inside a JSON blob
"""

import os
import json
import numpy as np


def get_json(file_name='../dxf.json.template'):
    """
    Load the JSON blob into native objects
    """
    with open(file_name, 'r') as f:
        t = json.load(f)
    return t


def write_json(template, file_name='../dxf.json.template'):
    """
    Write a native object to a JSON blob
    """
    with open(file_name, 'w') as f:
        json.dump(template, f, indent=4)


def replace_whitespace(text, SAFE_SPACE='|<^>|', insert=True):
    """
    Replace non-strippable whitepace in a string with a safe space
    """
    if insert:
        # replace whitespace with safe space chr
        args = (' ', SAFE_SPACE)
    else:
        # replace safe space chr with whitespace
        args = (SAFE_SPACE, ' ')
    lines = [line.strip().replace(*args)
             for line in str.splitlines(text)]
    # remove any blank lines
    if any(len(L) == 0 for L in lines):
        shaped = np.reshape(lines, (-1, 2))
        mask = np.ones(len(shaped), dtype=bool)
        for i, v in enumerate(shaped[:, 1]):
            if len(v) == 0:
                mask[i] = False
        lines = shaped[mask].ravel()
    return '\n'.join(lines)


def write_files(template, destination='./dxf'):
    """
    For a dict, write each value to destination/key
    """
    os.makedirs(destination)
    for key, value in template.items():
        with open(os.path.join(destination, key), 'w') as f:
            f.write(replace_whitespace(value, insert=False))


def read_files(path):
    """
    For a directory full of files, retrieve it
    as a dict with file_name:text
    """
    template = {}
    for file_name in os.listdir(path):
        # skip emacs buffers
        if '~' in file_name:
            continue
        with open(os.path.join(path, file_name), 'r') as f:
            template[file_name] = replace_whitespace(
                f.read(), insert=True)

    return template


if __name__ == '__main__':
    import sys

    # dump files to JSON template
    if 'dump_json' in sys.argv:
        t = read_files('dxf')
        write_json(t)
    elif 'read_json' in sys.argv:
        # dump JSON to files for editing
        t = get_json()
        write_files(t)
    else:
        print("run with 'read_json' to dump JSON to files")
        print("Or 'dump_json' to dump files to JSON")
