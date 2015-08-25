# conversions to inches
_TO_INCHES = {'microinches' : 1.0 / 1000.0, 
              'mils'        : 1.0 / 1000.0,
              'inches'      : 1.00,
              'feet'        : 12.0,
              'yards'       : 36.0,
              'miles'       : 63360,
              'angstroms'   : 1.0 / 2.54e8,
              'nanometers'  : 1.0 / 2.54e7,
              'microns'     : 1.0 / 2.54e4,
              'millimeters' : 1.0 / 2.54e1, 
              'centimeters' : 1.0 / 2.54e0,
              'meters'      : 1.0 / 2.54e-2,
              'kilometers'  : 1.0 / 2.54e-5,
              'decimeters'  : 1.0 / 2.54e-1,
              'decameters'  : 1.0 / 2.54e-3,
              'hectometers' : 1.0 / 2.54e-4,
              'gigameters'  : 1.0 / 2.54e-11,
              'AU'          : 5889679948818.897,
              'light years' : 3.72461748e17,
              'parsecs'     : 1.21483369e18}

def unit_conversion(current, desired):
    conversion = _TO_INCHES[current] / _TO_INCHES[desired]
    return conversion

def unit_guess(scale):
    '''
    Wild ass guess for the units of a drawing or model, based on the scale.
    '''
    if scale > 100.0: 
        return 'millimeters'
    else:      
        return 'inches'
