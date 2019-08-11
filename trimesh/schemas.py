"""
schemas.py
-------------

Tools for dealing with schemas, particularly JSONschema
"""
import json


def resolve(schema, resolver, recursive=True):
    """
    Given a JSON schema that references other files by URI
    replace them and return a flat version of the schema.

    Parameters
    ------------
    schema : dict or list
      JSON serializable JSONschema
    resolver : trimesh.resolvers.Resolver
      A way to load referenced resourced
    recursive : bool
      Recursively evaluate references

    Returns
    ------------
    flat : dict
      Schema with references replaced
    """
    pass


def resolve_json(text, resolver, recursive=True, fill_empty='{}'):
    """
    Given a JSON Schema containing `$ref` keys, replace all
    referenced URI values with their values using trimesh
    Resolver objects.


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

    assert '$ref' not in current
    # any values after the last reference
    result.append(current)

    # append results into single string
    appended = ' '.join(result)

    # assert we got rid of all the references
    assert '$ref' not in appended

    return appended
