"""
units.py
--------------
Deal with physical unit systems (i.e. inches, mm)

Very basic conversions, and no requirement for
sympy.physics.units or pint.
"""
import json

from .constants import log
from .resources import get_resource

# scaling factors from various unit systems to inches
TO_INCH = json.loads(get_resource('units_to_inches.json'))


def unit_conversion(current, desired):
    """
    Calculate the conversion from one set of units to another.

    Parameters
    ---------
    current : str
        Unit system values are in now (eg 'millimeters')
    desired : str
        Unit system we'd like values in (eg 'inches')

    Returns
    ---------
    conversion : float
        Number to multiply by to put values into desired units
    """
    current = str(current).strip().lower()
    desired = str(desired).strip().lower()
    conversion = TO_INCH[current] / TO_INCH[desired]
    return conversion


def units_from_metadata(obj, guess=True):
    """
    Try to extract hints from metadata and if that fails
    guess based on the object scale.


    Parameters
    ------------
    obj: object
        Has attributes 'metadata' (dict) and 'scale' (float)
    guess : bool
        If metadata doesn't indicate units, guess from scale

    Returns
    ------------
    units: str
        A guess of what the units might be
    """
    # try to guess from metadata
    for key in ['file_name', 'name']:
        if key not in obj.metadata:
            continue
        # get the string which might contain unit hints
        hints = obj.metadata[key].lower()
        if 'unit' in hints:
            # replace all delimiter options with white space
            for delim in '_-.':
                hints = hints.replace(delim, ' ')
            # loop through each hint
            for hint in hints.strip().split():
                # key word is "unit" or "units"
                if 'unit' not in hint:
                    continue
                # get rid of keyword and whitespace
                hint = hint.replace(
                    'units', '').replace(
                        'unit', '').strip()
                # if the hint is a valid unit return it
                if hint in TO_INCH:
                    return hint

    if not guess:
        raise ValueError('no units and not allowed to guess')

    # we made it to the wild ass guess section
    # if the scale is larger than 100 mystery units
    # declare the model to be millimeters, otherwise inches
    log.warning('no units: guessing from scale')
    if float(obj.scale) > 100.0:
        return 'millimeters'
    else:
        return 'inches'


def _convert_units(obj, desired, guess=False):
    """
    Given an object with scale and units try to scale
    to different units via the object's `apply_scale`.

    Parameters
    ---------
    obj :  object
        With apply_scale method (i.e. Trimesh, Path2D, etc)
    desired : str
        Units desired (eg 'inches')
    guess:   bool
        Whether we are allowed to guess the units
        if they are not specified.
    """
    if obj.units is None:
        # try to extract units from metadata
        # if nothing specified in metadata and not allowed
        # to guess will raise a ValueError
        obj.units = units_from_metadata(obj, guess=guess)

    log.info('converting units from %s to %s', obj.units, desired)
    # float, conversion factor
    conversion = unit_conversion(obj.units, desired)

    # apply scale uses transforms which preserve
    # cached properties (rather than just multiplying vertices)
    obj.apply_scale(conversion)
    # units are now desired units
    obj.units = desired
