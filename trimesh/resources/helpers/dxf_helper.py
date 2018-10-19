"""
dxf_helper.py
---------------

Manipulate DXF templates as plain text files rather
than strings inside a JSON blob
"""

import os
import json


def get_json(file_name='../dxf.json.template'):
    with open(file_name, 'r') as f:
        t = json.load(f)
    return t


def write_json(template, file_name='../dxf.json.template'):
    with open(file_name, 'w') as f:
        json.dump(template, f, indent=4)


def write_files(template, destination='./dxf'):
    os.makedirs(destination)
    for key, value in template.items():
        with open(os.path.join(destination, key), 'w') as f:
            f.write(value)


def read_files(path):
    template = {}
    for file_name in os.listdir(path):
        with open(os.path.join(path, file_name), 'r') as f:
            template[file_name] = f.read()

    return template


if __name__ == '__main__':
    # dump files to JSON template
    t = read_files('dxf')
    write_json(t)

    # dump JSON to files for editing
    #$t = get_json()
    #$write_files(t)
