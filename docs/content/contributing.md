Contributing To Trimesh
=======================

Pull requests are always super welcome! Trimesh is a relatively small open source project and really benefits from the bugfixes, features, and other stuff the 200+ contributors have PR'd, so thanks!


## Developer Quick Start

Here's how I set up a new environment and write functions. It's not necessary to do it this way but it does make some things easier. If you're planning on editing trimesh you might want to fork it via the Github interface, then install it via an editable pip install:
```
# you probably want to clone your fork
git clone git@github.com:mikedh/trimesh.git

# do an editable install so you can experiment
cd trimesh
uv sync --extra all
```


I pretty much always start with an interactive terminal (i.e. a "REPL") inside a stub function:
```
import trimesh
import numpy as np

def fancy_function(blah):
    if blah.shape != (3, 3):
       raise ValueError('this input was goofy!')

    # do some obvious operations and whatnot
    dots = np.dot(blah, [1,2,3])

    # get a REPL inside my function so I can write each line
    # with the context of the function, copy paste the lines
    # in and at the end return the value and remove the embed
    from IPython import embed
    embed()
    
if __name__ == '__main__':
    # print out all the debug messages so we can see
    # if there's something going on we need to look at
    trimesh.util.attach_to_log()

    # I like pyinstrument as it's a relatively low-overhead sampling
    # profiler and has nice looking nested print statements compared
    # to cProfile or others.
    import pyinstrument
    
    data = np.random.random((3, 3))
    with pyinstrument.Profiler() as pr:
        result = fancy_function(data)
    pr.print()
```

When you remove the embed and see the profile result you can then tweak the lines that are slow before finishing the function.

### Automatic Formatting
Trimesh uses `ruff` for both linting and formatting which is configured in `pyproject.toml`, you can run with:
```
ruff check --fix
ruff format
```

## Docstrings

Trimesh uses the [Sphinx Numpy-style](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy) docstrings which get parsed into the API reference page. 


## Deprecations

We try to add a somewhat helpful `DeprecationWarning` one year in advance of a major API change:

```python
    warnings.warn(
        "`remove_duplicate_faces` is deprecated "
        + "and will be removed in March 2024: "
            + "replace with `mesh.update_faces(mesh.unique_faces())`",
        category=DeprecationWarning,
        stacklevel=2,
    )
```

## Pull Requests And AI

Contributions are welcome! There's an AI policy in the repository root aimed at reducing the effort for maintainers.

The most compelling issue or PR has a ~10-line standalone reproduction containing one or more strict `assert condition` that fails in the `main` branch. Adding test models is OK but undesirable. After the error reliably fails the assertion, try to keep diffs of fixes small and surgical. Assertion quality is going to be the main review point. Think about the STRONGEST possible predicate that an `assert` gives full confidence in the solution, and that maintainers can't break in the future, and makes cheating the test harder than root-cause fixes. 

For example, if you were writing a 2D convex hulls algorithm: if you traverse hull segments in order, the projection for every vertex against signed perpendicular vector should be between negative infinity and floating-point epsilon. This one assertion guarantees consistent winding (through the signed vectors), AND convexity of the result, and is simple, easy to calculate, and catches nearly all errors. Some other predicates that you can check against truth that can be useful are `mesh.is_volume`, `mesh.area`, `mesh.volume`, but the stronger the better!

Mesh analysis in Python is quite performance sensitive, and very minor changes can make things unusably slow. A PR is more compelling if it has a profiler output (`pyinstrument`) for every use of the old code in unit tests, and the matching profile for the new code running on everything in the unit tests. If the new thing is 10x slower than the old thing, it is bad.


## General Tips

Python can be fast but only when you lean heavily on `numpy`. In general, if you ever have a block which loops through faces and vertices it will be basically unusable with even moderately sized meshes. All operations on face or vertex arrays should be vectorized numpy operations unless absolutely unavoidable. Profiling helps figure out what is slow, but some general advice:
