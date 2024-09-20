import argparse


def main():
    """
    A simple command line utility for accessing trimesh functions.

    To display a mesh:
      > trimesh hi.stl

    To convert a mesh:
      > trimesh hi.stl -e hey.glb

    To print some information about a mesh:
      > trimesh hi.stl --statistics
    """
    from .exchange.load import load

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name")
    args = parser.parse_args()
    load(args.filename).show()


if __name__ == "__main__":
    main()
