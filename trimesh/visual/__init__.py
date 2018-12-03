"""
visual
-------------

Provide
"""

from .color import (ColorVisuals,
                    random_color,
                    to_rgba,
                    create_visual,
                    interpolate,
                    linear_color_map)

# explicitly list imports in __all__
# as otherwise flake8 gets mad
__all__ = [ColorVisuals,
           random_color,
           to_rgba,
           create_visual,
           interpolate,
           linear_color_map]
