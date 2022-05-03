Contributing To Trimesh
=======================

Pull requests are always super welcome! Trimesh is a relatively small open source project and really benefits from the bugfixes, features, and other stuff the 100+ contributors have PR'd, so thanks!


## Developer Quick Start

Here's how I set up a new environment and write functions. It's not necessary to do it this way but it does make some things easier! If you don't have a "virtual environment" solution there's plenty of ways to do this (poetry, pipenv, conda, etc.) but I just use the `venv` module built into the standard library:
```
# create the venv
python -m venv ~/myenv

# on linux this will use this venv
# every time you open a new terminal
echo "source ~/myenv/bin/activate" >> ~/.bashrc
```

With a virtual environment so pip doesn't dump files everywhere you can install some stuff. The `flake8-no-implicit-concat` adds a rule which disallows strings on different lines with no comma being concatenated as this is pretty much always a bug.
```
pip install autopep8 flake8 flake8-no-implicit-concat codespell pyinstrument ipython
```

I pretty much always start with an interactive terminal (i.e. a "REPL") inside a stub function:
```
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
Before opening a pull request I run some auto-formatting rules which will run autopep8 and yell at you about any flake8 rule violations. There is a convenience script baked into `setup.py` to run all of these which you can run with:
```
python setup.py --format
```

This is equivalent to running `codespell`, `autopep8`, and `flake8` on trimesh, examples, and tests. You can also run it yourself with these options:
```
autopep8 --recursive --verbose --in-place --aggressive trimesh
flake8 trimesh
```

## General Tips

Python can be fast but only when you use it as little as possible. In general, if you ever have a block which loops through faces and vertices it will be basically unusable with even moderately sized meshes. All operations on face or vertex arrays should be vectorized numpy operations unless absolutely unavoidable. Profiling helps figure out what is slow, but some general advice:

### Helpful
- Run your test script with `ipython -i newstuff.py` and profile with magic, i.e. `%timeit var.split()`
- Use `np.fromstring`, `np.frombuffer`
- Use `str.split` or `np.fromstring`:
```
In [6]: %timeit np.array(text.split(), dtype=np.float64)
1000 loops, best of 3: 209 µs per loop

In [7]: %timeit np.fromstring(text, sep='\n', dtype=np.float64)
10000 loops, best of 3: 139 µs per loop
```
- Use giant format strings rather than looping, appending, or even iterator joining:
```
In [14]: array = np.random.random((10000,3))

In [15]: %timeit '\n'.join('{}/{}/{}'.format(*row) for row in array)
10 loops, best of 3: 60.3 ms per loop

In [16]: %timeit ('{}/{}/{}\n' * len(array))[:-1].format(*array.flatten())
10 loops, best of 3: 34.3 ms per loop
```
- Sometimes you can use sparse matrices to replace a loop and get a huge speedup. [Here's the same algorithm implemented two ways, looping and sparse dot products.](https://github.com/mikedh/trimesh/blob/master/trimesh/geometry.py#L186-L203)
- In tight loops, `array.sum(axis=1)` often pops up as the slowest thing. This can be replaced with a dot product of ones, which are very optimized can be substantially faster:
```
In [1]: import numpy as np

In [2]: a = np.random.random((10000, 3))

In [3]: %timeit a.sum(axis=1)
10000 loops, best of 3: 157 µs per loop

In [4]: %timeit np.dot(a, [1,1,1])
10000 loops, best of 3: 25.8 µs per loop
```
- If you can use it, `np.concatenate` is usually faster than `np.vstack`, `np.append`, or `np.column_stack`
```
In [3]: seq = [np.random.random((int(np.random.random() * 1000), 3)) for i in range(1000)]

In [7]: %timeit np.vstack(seq)
100 loops, best of 3: 3.48 ms per loop

In [8]: %timeit np.concatenate(seq)
100 loops, best of 3: 2.33 ms per loop
```
- Sometimes `np.bincount` can be used instead of `np.unique` for a substantial speedup:
```
In [45]: a = (np.random.random(1000) * 1000).astype(int)

In [46]: set(np.where(np.bincount(a).astype(bool))[0]) == set(np.unique(a))
Out[46]: True

In [47]: %timeit np.where(np.bincount(a).astype(bool))[0]
100000 loops, best of 3: 5.81 µs per loop

In [48]: %timeit np.unique(a)
10000 loops, best of 3: 31.8 µs per loop
```

### Try To Avoid
- Looping in general, and *especially* looping on arrays that could have many elements(i.e. vertices and faces). The loop overhead is very high in Python. If necessary to loop you may find that list comprehensions are quite a bit faster (though definitely profile first!) probably for scoping reasons.
- Boolean operations (i.e. intersection, difference, union) on meshes may seem like the answer, but they are nearly always flaky and slow. The best answer is usually to restructure your problem to use some form of vector checks if possible (i.e. dot products, ray tests, etc). Look at `trimesh.intersections` for an example of a problem that could have used a boolean operation but didn't.
