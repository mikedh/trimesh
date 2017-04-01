from .constants import log

# conversions to inches
_TO_INCHES = {'microinches': 1.0 / 1000.0,
              'mils': 1.0 / 1000.0,
              'inches': 1.00,
              'feet': 12.0,
              'yards': 36.0,
              'miles': 63360,
              'angstroms': 1.0 / 2.54e8,
              'nanometers': 1.0 / 2.54e7,
              'microns': 1.0 / 2.54e4,
              'millimeters': 1.0 / 2.54e1,
              'centimeters': 1.0 / 2.54e0,
              'meters': 1.0 / 2.54e-2,
              'kilometers': 1.0 / 2.54e-5,
              'decimeters': 1.0 / 2.54e-1,
              'decameters': 1.0 / 2.54e-3,
              'hectometers': 1.0 / 2.54e-4,
              'gigameters': 1.0 / 2.54e-11,
              'AU': 5889679948818.897,
              'light years': 3.72461748e17,
              'parsecs': 1.21483369e18}

# if a unit is known by other symbols, include them here
_synonyms = {'millimeters': ['mm'],
             'inches': ['in'],
             'meters': ['m']}

for key, new_keys in _synonyms.items():
    _value = _TO_INCHES[key]
    for new_key in new_keys:
        _TO_INCHES[new_key] = _value


def unit_conversion(current, desired):
    '''
    Calculate the conversion from one set of units to another.

    Parameters
    ---------
    current: str, unit system values are in now (eg 'millimeters')
    desired: str, unit system we'd like values in (eg 'inches')

    Returns
    ---------
    conversion: float, number to multiply by to put values into desired units
    '''
    conversion = _TO_INCHES[current] / _TO_INCHES[desired]
    return conversion


def validate(units):
    '''
    Check whether a string represents the name of a valid unit

    Returns
    ---------
    valid: bool, is units string a valid unit or not
    '''
    valid = str(units) in _TO_INCHES
    return valid


def unit_guess(scale):
    '''
    Wild ass guess for the units of a drawing or model, based on the scale.
    '''
    if scale > 100.0:
        return 'millimeters'
    else:
        return 'inches'


def _set_units(obj, desired, guess):
    '''
    Given an object that has units and vertices attributes convert units.

    Parameters
    ---------
    obj:     object with units and vertices (eg Path or Trimesh)
    desired: units desired (eg 'inches')
    guess:   boolean, whether we are allowed to guess the units of the document
             if they are not specified.
    '''
    desired = str(desired)
    if not validate(desired):
        raise ValueError(desired + ' are not a valid unit!')

    if obj.units is None:
        if guess:
            obj.units = unit_guess(obj.scale)
            log.warning('No units specified, guessing units are %s',
                        obj.units)
        else:
            raise ValueError('No units specified and not allowed to guess!')
    log.info('Converting units from %s to %s', obj.units, desired)
    conversion = unit_conversion(obj.units, desired)
    obj.vertices *= conversion
    obj.units = desired
