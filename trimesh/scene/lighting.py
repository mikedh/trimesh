"""
light.py
--------------

Hold basic information about lights, forked from pyrender:
https://raw.githubusercontent.com/mmatl/pyrender/master/pyrender/light.py"""

import abc
import sys
import numpy as np

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Light(ABC):
    """
    Base class for all light objects.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the range is assumed to be infinite.
    """

    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None):
        self.name = name
        self.color = color
        self.intensity = intensity
        self.range = range

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = format_color_vector(value, 3)

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = float(value)

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        if value is None or value < 0.0:
            self._range = value
        else:
            self._range = float(value)


class DirectionalLight(Light):
    """
    Directional lights are light sources that act as though they are
    infinitely far away and emit light in the direction of the local -z axis.
    This light type inherits the orientation of the node that it belongs to;
    position and scale are ignored except for their effect on the inherited
    node orientation. Because it is at an infinite distance, the light is
    not attenuated. Its intensity is defined in lumens per metre squared,
    or lux (lm/m2).

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the range is assumed to be infinite.
    """

    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None):
        super(DirectionalLight, self).__init__(
            name=name,
            color=color,
            intensity=intensity,
            range=range
        )


class PointLight(Light):
    """
    Point lights emit light in all directions from their position in space;
    rotation and scale are ignored except for their effect on the inherited
    node position. The brightness of the light attenuates in a physically
    correct manner as distance increases from the light's position (i.e.
    brightness goes like the inverse square of the distance). Point light
    intensity is defined in candela, which is lumens per square radian (lm/sr).

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the range is assumed to be infinite.
    """

    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None):
        super(PointLight, self).__init__(
            name=name,
            color=color,
            intensity=intensity,
            range=range
        )


class SpotLight(Light):
    """
    Spot lights emit light in a cone in the direction of the local -z axis.
    The angle and falloff of the cone is defined using two numbers, the
    `innerConeAngle` and `outerConeAngle`. As with point lights, the brightness
    also attenuates in a physically correct manner as distance increases from
    the light's position (i.e. brightness goes like the inverse square of the
    distance). Spot light intensity refers to the brightness inside the
    `innerConeAngle` (and at the location of the light) and is defined in
    candela, which is lumens per square radian (lm/sr). A spot light's position
    and orientation are inherited from its node transform. Inherited scale does
    not affect cone shape, and is ignored except for its effect on position
    and orientation.

    Attributes
    ----------
    name : str, optional
        Name of the light.
    color : (3,) float
        RGB value for the light's color in linear space.
    intensity : float
        Brightness of light. The units that this is defined in depend on the type of light.
        point and spot lights use luminous intensity in candela (lm/sr),
        while directional lights use illuminance in lux (lm/m2).
    range : float
        Cutoff distance at which light's intensity may be considered to
        have reached zero. Supported only for point and spot lights, must be > 0.
        If None, the range is assumed to be infinite.
    innerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff begins.
        Must be greater than or equal to `0` and less than `outerConeAngle`.
    outerConeAngle : float
        Angle, in radians, from centre of spotlight where falloff ends.
        Must be greater than `innerConeAngle` and less than or equal to `PI / 2.0`.
    """

    def __init__(self,
                 name=None,
                 color=None,
                 intensity=None,
                 range=None,
                 innerConeAngle=0.0,
                 outerConeAngle=np.pi / 4.0):
        super(SpotLight, self).__init__(
            name=name,
            color=color,
            intensity=intensity,
            range=range
        )
        self.outerConeAngle = outerConeAngle
        self.innerConeAngle = innerConeAngle

    @property
    def innerConeAngle(self):
        return self._innerConeAngle

    @innerConeAngle.setter
    def innerConeAngle(self, value):
        if value < 0.0 or value > self.outerConeAngle:
            raise ValueError('Invalid value for inner cone angle')
        self._innerConeAngle = float(value)

    @property
    def outerConeAngle(self):
        return self._outerConeAngle

    @outerConeAngle.setter
    def outerConeAngle(self, value):
        if value < 0.0 or value > np.pi / 2.0 + 1e-9:
            raise ValueError('Invalid value for outer cone angle')
        self._outerConeAngle = float(value)
