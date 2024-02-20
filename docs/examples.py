"""
examples.py
------------

Convert `ipynb` to a web-renderable format from the contents
of `../examples/*.ipynb`
"""

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
{title} </{file_name}.html>
"""


def generate_index(source: str, target: str) -> str:
    """
    Go through a directory of source `ipynb` files and write
    an RST index with a toctree.

    Also postprocesses the results of `jupyter nbconvert`
    """

    lines = [
        "Examples",
        "===========",
        "Several examples are available as rendered IPython notebooks.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]

    target_dir = os.path.dirname(target)

    for fn in os.listdir(source):
        if not fn.lower().endswith(".ipynb"):
            continue

        name = fn.rsplit(".")[0]
        title = name.replace("_", " ").title()
        # notebook converted to RST
        convert = os.path.join(target_dir, f"{name}.rst")
        if not os.path.exists(convert):
            print(f"no RST for {name}.rst")
            continue

        with open(convert) as f:
            doc, post = postprocess(f.read(), title=title)
        with open(convert, "w") as f:
            f.write(post)

        lines.append(f"   {name}")
        # lines.append(doc)
        lines.append("")

    return "\n".join(lines)


def postprocess(text: str, title: str) -> str:
    """
    Postprocess an RST generated from `jupyter nbconvert`
    """
    lines = str.splitlines(text)

    # already has a title so exit
    if "===" in "".join(lines[:4]):
        return "", text

    head = []
    index = 0
    ready = False
    for i, L in enumerate(lines):
        if "parsed-literal" in L:
            ready = True
            continue
        if ready:
            if "code::" in L:
                index = i
                break
            else:
                head.append(L)

    # clean up the "parsed literal"
    docstring = (
        " ".join(" ".join(head).replace("\\n", " ").split()).strip().strip("'").strip()
    )

    # add a title and the docstring as a header
    clip = f"{title}\n=============\n{docstring}\n\n" + "\n".join(lines[index:])

    return docstring, clip


if __name__ == "__main__":
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

    with open(target, "w") as f:
        f.write(generate_index(source=source, target=target))
