trimesh.resources.javascript
-------------

The HTML/JS for the jupyter notebook viewer. If you edit `load_base64.js` and `viewer.html` to do what you want them to do, you can then run `python compile.py` and the template JSON blob will be created and minified. The test data in `viewer.html` will be replaced with a template string.

First fetch the files:
```
bash update.bash
```

Then edit `viewer.html` to work. When you're done, run the artifact compile step:
```
python compile.py
```