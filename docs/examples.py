"""
examples.py
------------

Generate `examples.md` from the contents
of `../examples/*.ipynb`
"""

import json
import logging
import os
import sys

log = logging.getLogger("trimesh")
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.DEBUG)

# current working directory
pwd = os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))


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

    source = loaded["cells"][0]["source"]

    assert source[0].strip() == '"""'
    assert source[-1].strip() == '"""'

    return " ".join(i.strip() for i in source[1:-1])


base = """
{title}
==========
.. toctree::
   :maxdepth: 2

   {file_name}
"""

if __name__ == "__main__":
    import argparse

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, help="a directory containing `ipynb` files", required=True
    )
    parser.add_argument(
        "--target", type=str, help="Where the generated .rst file goes", required=True
    )
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    target = os.path.abspath(args.target)

    markdown = [
        "# Examples",
        "===========",
        "Several examples are available as rendered IPython notebooks.",
        "",
    ]

    for fn in os.listdir(source):
        if not fn.lower().endswith(".ipynb"):
            continue
        path = os.path.join(source, fn)
        with open(path) as f:
            raw = json.load(f)
        doc = extract_docstring(raw)

        #
        name = fn.split(".")[0]
        file_name = f"examples.{name}.rst"
        title = " ".join(name.split("_")).title()
        markdown.append(base.format(title=title, file_name=file_name))

    final = "\n".join(markdown)
    with open(target, "w") as f:
        f.write(final)
