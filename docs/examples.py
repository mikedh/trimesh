"""
examples.py
------------

Generate `examples.md` from the contents
of `../examples/*.ipynb`
"""

import os
import sys
import json
import logging

log = logging.getLogger('trimesh')
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)

# current working directory
pwd = os.path.abspath(os.path.expanduser(
    os.path.dirname(__file__)))

# where are our notebooks to render
source = os.path.abspath(os.path.join(
    pwd, '..', 'examples'))

# which index file are we generating
target = os.path.abspath(os.path.join(
    pwd, "examples.md"))


def extract_docstring(loaded):
    """
    Given a loaded JSON `ipynb` notebook extract
    the docstring from the first cell and error
    if the first cell doesn't have a docstring.

    Parameters
    ------------
    loaded : dict
      Loaded ipynb format.

    Returns
    ----------
    doc : str
      Cleaned up docstring.
    """

    source = loaded['cells'][0]['source']

    assert source[0].strip() == '"""'
    assert source[-1].strip() == '"""'

    return ' '.join(i.strip() for i in source[1:-1])


if __name__ == '__main__':

    markdown = ['# Examples',
                'Several examples are available as rendered IPython notebooks.', '', ]

    for fn in os.listdir(source):
        if not fn.lower().endswith('.ipynb'):
            continue
        path = os.path.join(source, fn)
        with open(path, 'r') as f:
            raw = json.load(f)
        doc = extract_docstring(raw)
        log.info('`{}`: "{}"\n'.format(fn, doc))
        link = f'examples.{fn.split(".")[0]}.html'

        markdown.append(f'### [{fn}]({link})')
        markdown.append(doc)
        markdown.append('')

    final = '\n'.join(markdown)
    with open(target, 'w') as f:
        f.write(final)
