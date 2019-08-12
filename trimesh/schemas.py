"""
schemas.py
-------------

Tools for dealing with schemas, particularly JSONschema
"""
import json


def resolve_json(text, resolver, recursive=True, fill_empty='{}'):
    """
    Given a JSON Schema containing `$ref` keys, replace all
    referenced URI values with their values using trimesh
    Resolver objects.

    Parameters
    ---------------
    text : str
      JSON text including `$ref` to other files
    resolver : trimesh.visual.resolver.Resolver
      Resolver to fetch referenced assets
    recursive : bool
      If True, resolve references in referenced files
    fill_empty : str
      What to replace empty references with

    Returns
    ----------
    result : str
      JSON text with references filled in
    """
    if '$ref' not in text:
        return text
    # append strings to list then concatenate once
    result = []
    # set the current value to the input text
    current = text
    # loop with a for to cap iterations
    for i in range(len(text)):
        # find the referenced key in the text
        idx = current.find('$ref')
        # if the string wasn't found find will return -1
        if idx < 0:
            break
        # find the first opening bracket before the ref key
        first = current.rfind('{', 0, idx)
        # find the first closing bracket after the ref key
        last = current.find('}', idx)

        # extract the URI of the reference
        name = json.loads(current[first:last + 1])['$ref']

        # get the bytes from the resolver
        data = resolver.archive[name].read().decode('utf-8')

        # include all text before the first reference
        result.append(current[:first])

        if recursive:
            result.append(
                resolve_json(
                    data, resolver=resolver, recursive=True))
        else:
            # include the loaded data
            result.append(data)

        # if we had a reference to an empty file fill it
        if len(result[-1].strip()) == 0:
            result.append(fill_empty)

        # for the next loop only look at the next chunk
        current = current[last + 1:]

    # any values after the last reference
    result.append(current)

    # append results into single string
    appended = ' '.join(result)

    # assert we got rid of all the references
    assert '$ref' not in appended

    return appended
