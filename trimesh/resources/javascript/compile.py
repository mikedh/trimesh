"""
compile.py
--------------

Take an HTML file and embed local scripts into one blob.

The idea is you develop on viewer.html, then call this file to
generate the template used in the trimesh viewer.
"""
import os
import jsmin
import requests

from lxml import html


def minify(path):
    """
    Load a javascript file and minify.

    Parameters
    ------------
    path: str, path of resource
    """

    if path.startswith('http'):
        data = requests.get(path).content.decode(
            'ascii', errors='ignore')
        print('downloaded', path, len(data))  # noqa
    else:
        with open(path, 'rb') as f:
            # some upstream JS uses unicode spaces -_-
            data = f.read().decode('ascii', errors='ignore')
    # don't re-minify
    if '.min.' in path:
        return data

    try:
        return jsmin.jsmin(data)
    except BaseException:
        return data


if __name__ == '__main__':
    # we're going to embed every non-CDN'd file
    h = html.parse('viewer.html')
    collection = []

    # find all scripts in the document
    for s in h.findall('//script'):
        if 'src' in s.attrib:
            if 'http' in s.attrib['src']:
                # pass # download CDN files and embed
                continue  # leave any remote files alone

            # get a blob of file
            path = s.attrib['src'].strip()
            print('minifying:', path)  # noqa
            mini = minify(path)

            # replace test data in our file
            if path == 'load_base64.js':
                print('replacing test data with "$B64GLTF"')  # noqa
                start = mini.find('base64_data')
                end = mini.find(';', start)
                # replace test data with a string we can replace
                # keep in quotes to avoid being minified
                mini = mini.replace(mini[start:end],
                                    'base64_data="$B64GLTF";')
            collection.append(mini)
        # remove the script reference
        s.getparent().remove(s)

    # a new script element with everything blobbed together
    ns = html.Element('script')
    ns.text = ''.join(collection)

    # append the new script element
    body = h.find('body')
    body.append(ns)

    result = html.tostring(h, pretty_print=False).decode('utf-8')
    # result = result.replace('<body>', '').replace('</body>', '')

    with open('../viewer.html.template', 'w') as f:
        f.write(result)

    import subprocess
    subprocess.check_call(['zip', '-9', '-j', '../viewer.template.zip',
                           '../viewer.html.template'])
    os.remove('../viewer.html.template')
