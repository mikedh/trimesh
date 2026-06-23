"""
Render a self-hosted SVG coverage badge from a percentage
"""

import sys

# (lower-bound percent, hex color) checked high-to-low
COLORS = [
    (90, "#4c1"),  # brightgreen
    (75, "#97ca00"),  # green
    (50, "#dfb317"),  # yellow
    (0, "#e05d44"),  # red
]

# shields-style template
TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" role="img" aria-label="coverage: {pct}">
<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>
<clipPath id="r"><rect width="{total}" height="20" rx="3" fill="#fff"/></clipPath>
<g clip-path="url(#r)">
<rect width="{left}" height="20" fill="#555"/>
<rect x="{left}" width="{right}" height="20" fill="{color}"/>
<rect width="{total}" height="20" fill="url(#s)"/>
</g>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="110" text-rendering="geometricPrecision">
<text x="{lx}" y="150" transform="scale(.1)" fill="#fff">coverage</text>
<text x="{rx}" y="150" transform="scale(.1)" fill="#fff">{pct}</text>
</g>
</svg>
"""


def color_for(percent):
    # first threshold the percent meets or exceeds
    for bound, hexcolor in COLORS:
        if percent >= bound:
            return hexcolor
    return COLORS[-1][1]


def render(total):
    # `total` is the string coverage already rounded, e.g. "86.9"
    pct = f"{total}%"
    # match shields.io geometry: 20px tall, ~7px per glyph at 11px Verdana
    # plus equal padding so this sits flush with the other README badges
    left = 61  # width of the "coverage" label, same as shields
    right = 7 * len(pct) + 12
    total_width = left + right
    return TEMPLATE.format(
        total=total_width,
        left=left,
        right=right,
        color=color_for(float(total)),
        pct=pct,
        lx=left * 5,
        rx=left * 10 + right * 5,
    )


if __name__ == "__main__":
    total, target = sys.argv[1], sys.argv[2]
    with open(target, "w") as f:
        f.write(render(total))
