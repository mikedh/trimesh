trimesh.io
==========

Mesh importers and exporters.


## Tips for New Importers/Exporters/Code
Python can be fast, but only when you use it as little as possible. In general, if you ever have a block which loops through faces and vertices it will be basically unusable with even moderately sized meshes. All operations on face or vertex arrays should be vectorized numpy operations unless absolutely unavoidable. Profiling helps figure out what is slow, but some general advice:

### Do
- Use `np.fromstring`, `np.frombuffer`
- Use `str.split`
- Run your test script with `ipython -i newstuff.py` and profile with magic, ie `%timeit var.split()`
- Use giant format strings rather than looping and appending
- Sometimes you can use sparse matrices to replace a loop for a fun 10,000x speedup


### Don't
- Loop through potentially giant arrays
