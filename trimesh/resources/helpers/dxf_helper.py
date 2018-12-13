"""
dxf_helper.py
---------------

Manipulate DXF templates as plain text files rather
than strings inside a JSON blob
"""

import os
import json


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

    return '\n'.join(line.strip().replace(*args)
                     for line in str.splitlines(text))


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
        with open(os.path.join(path, file_name), 'r') as f:
            template[file_name] = replace_whitespace(f.read(), insert=True)

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
        print("run with 'read_json' to dump JSON to files or 'dump_json' to dump files to JSON")
