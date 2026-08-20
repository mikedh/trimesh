"""
animation.py
--------------

Keyframed animation of a scene, stored as translation,
rotation, and scale sampled at increasing times.
"""

import numpy as np

from .. import caching
from ..transformations import quaternion_slerp, tqs_from_matrix, tqs_matrix
from ..typed import Floating, Hashable, NDArray1D, NDArray3D, float64

# how values between keyframes are computed, which map onto the
# GLTF sampler interpolation modes of LINEAR, STEP, and CUBICSPLINE
_INTERPOLATION = ("linear", "step", "cubic")

# the three channels of a keyframe, in the order a TQS tuple holds them
_FIELDS = ("translation", "quaternion", "scale")

# one keyframe of a node transform: a time, the transform decomposed into
# translation-quaternion-scale, and the cubic tangents on either side of it
# the tangents are zero unless `interpolation` is "cubic"
KEYFRAME = np.dtype(
    [
        ("time", float64),
        ("translation", float64, 3),
        ("quaternion", float64, 4),
        ("scale", float64, 3),
        ("translation_in", float64, 3),
        ("quaternion_in", float64, 4),
        ("scale_in", float64, 3),
        ("translation_out", float64, 3),
        ("quaternion_out", float64, 4),
        ("scale_out", float64, 3),
    ]
)


def keyframes_from_matrix(times, matrices=None) -> NDArray1D:
    """
    Build a keyframe array from times and homogeneous transforms.

    Parameters
    ------------
    times : (n,) float
      Keyframe times in seconds.
    matrices : (n, 4, 4) float or None
      Transform at each time, None for an identity at every keyframe.

    Returns
    ----------
    keyframes : (n,) KEYFRAME
      Keyframes with zeroed cubic tangents.
    """
    times = np.asanyarray(times, dtype=np.float64).reshape(-1)

    keyframes = np.zeros(len(times), dtype=KEYFRAME)
    keyframes["time"] = times

    if matrices is None:
        # an all-zero keyframe isn't an identity transform
        keyframes["scale"] = 1.0
        keyframes["quaternion"][:, 0] = 1.0
        return keyframes

    (
        keyframes["translation"],
        keyframes["quaternion"],
        keyframes["scale"],
    ) = tqs_from_matrix(np.asanyarray(matrices, dtype=np.float64).reshape((-1, 4, 4)))

    return keyframes


def hermite(before, after, tangent_out, tangent_in, fraction, width):
    """
    Evaluate a cubic Hermite spline between bracketing keyframe values.

    Parameters
    ------------
    before, after : (n, d) float
      Value at the keyframe bracketing each query on either side.
    tangent_out, tangent_in : (n, d) float
      Outgoing tangent of `before` and incoming tangent of `after`.
    fraction : (n, 1) float
      How far between the two keyframes each query is.
    width : (n, 1) float
      Length of the interval: a tangent is a rate of change so it
      scales with the time it acts over.

    Returns
    ----------
    values : (n, d) float
      The curve evaluated at each query.
    """
    squared = fraction**2
    cubed = fraction**3

    # the basis is exactly `before` at a fraction of zero and `after` at one
    return (
        (2.0 * cubed - 3.0 * squared + 1.0) * before
        + width * (cubed - 2.0 * squared + fraction) * tangent_out
        + (-2.0 * cubed + 3.0 * squared) * after
        + width * (cubed - squared) * tangent_in
    )


class RigidAnimation:
    """
    Keyframed transforms driving one edge of a scene graph.

    An animation drives an edge rather than a node: the keyframes are
    the `frame_from -> frame_to` transform, which is what makes them
    compose with whatever the rest of the graph is doing above them.

    Animations which share a `name` are exported as a single
    glTF animation, which is how a multi-node motion is grouped.
    """

    def __init__(
        self,
        frame_to: Hashable,
        frame_from: Hashable = None,
        times=None,
        matrices=None,
        keyframes=None,
        name: str = "animation",
        interpolation: str = "linear",
    ):
        """
        Create an animation from either matrices or a keyframe array.

        Parameters
        ------------
        frame_to : hashable
          The scene graph node this drives.
        frame_from : hashable or None
          Which node the keyframes are relative to, usually the parent of
          `frame_to`. None means the base frame, as in `SceneGraph.update`.
        times : (n,) float or None
          Keyframe times in seconds, increasing. Pass with `matrices`.
        matrices : (n, 4, 4) float or None
          The `frame_from -> frame_to` transform at each time. Note this is
          across one edge, not `scene.graph[frame_to]` which is from the base.
        keyframes : (n,) KEYFRAME or None
          Keyframes to use directly, instead of `times` and `matrices`.
        name : str
          Animations sharing a name export as one glTF animation.
        interpolation : str
          One of "linear", "step", or "cubic".

        Raises
        -----------
        ValueError
          If the keyframes don't correspond or are malformed.
        """
        self._data = caching.DataStore()
        self._cache = caching.Cache(id_function=self._data.__hash__)

        self.frame_to = frame_to
        self.frame_from = frame_from
        self.name = name
        self.interpolation = interpolation

        if keyframes is None:
            if times is None or matrices is None:
                raise ValueError("pass either `keyframes` or `times` and `matrices`!")
            if len(np.asanyarray(times).reshape(-1)) != len(
                np.asanyarray(matrices).reshape((-1, 4, 4))
            ):
                raise ValueError("times and matrices must correspond!")
            keyframes = keyframes_from_matrix(times, matrices)
        elif times is not None or matrices is not None:
            raise ValueError("pass `keyframes` or `times` and `matrices`, not both!")

        self.keyframes = keyframes

    @property
    def keyframes(self) -> NDArray1D:
        """
        Time, transform, and cubic tangents of each keyframe.

        Returns
        ----------
        keyframes : (n,) KEYFRAME
        """
        return self._data["keyframes"]

    @keyframes.setter
    def keyframes(self, values) -> None:
        keyframes = caching.tracked_array(values, dtype=KEYFRAME).reshape(-1)

        if len(keyframes) == 0:
            raise ValueError("animation must have at least one keyframe!")
        if (np.diff(keyframes["time"]) < 0).any():
            raise ValueError("times must be increasing!")

        self._data["keyframes"] = keyframes

    @property
    def times(self) -> NDArray1D[float64]:
        """
        When each keyframe occurs.

        Returns
        ----------
        times : (n,) float
          Keyframe times in seconds, increasing.
        """
        return self.keyframes["time"]

    @property
    def interpolation(self) -> str:
        """
        How values between keyframes are computed.

        Returns
        ----------
        interpolation : str
          One of "linear", "step", or "cubic".
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: str) -> None:
        cleaned = str(value).strip().lower()
        if cleaned not in _INTERPOLATION:
            raise ValueError(f"unsupported interpolation `{value}`!")
        self._interpolation = cleaned

    @property
    def duration(self) -> float:
        """
        How long this animation runs for.

        Note this is the time of the final keyframe rather than the span
        between the first and last, as that is what a viewer loops on.

        Returns
        ----------
        duration : float
          Length of the animation, in seconds.
        """
        return float(self.keyframes["time"][-1])

    @caching.cache_decorator
    def matrices(self) -> NDArray3D[float64]:
        """
        Local `frame_from -> frame_to` transform at each keyframe.

        Returns
        ----------
        matrices : (n, 4, 4) float
        """
        keyframes = self.keyframes
        return tqs_matrix(
            keyframes["translation"], keyframes["quaternion"], keyframes["scale"]
        )

    def at(self, time: Floating) -> NDArray3D[float64]:
        """
        Sample the transform across this edge at one or more times.

        Times outside the keyframe range clamp to the first or last
        keyframe rather than extrapolating.

        Parameters
        ------------
        time : float or (m,) float
          Times to sample at, in seconds.

        Returns
        ----------
        matrices : (4, 4) or (m, 4, 4) float
          Transform at each requested time.
        """
        query = np.array(time, dtype=np.float64)
        # only return a single matrix if a single time was passed
        single = query.ndim == 0
        query = query.reshape(-1)

        # as a plain array: indexing a tracked array yields a tracked
        # array, which then hashes itself every time it is used
        keyframes = np.asarray(self.keyframes)
        times = keyframes["time"]

        if len(keyframes) == 1:
            # a single keyframe is constant for all time, and would
            # otherwise leave no interval to bracket a query time with
            sampled = np.tile(self.matrices[0], (len(query), 1, 1))
        elif self.interpolation == "step":
            # a step holds the keyframe at-or-before the query time
            # note `side="right"` so landing exactly on a keyframe
            # picks that keyframe rather than the one before it
            sampled = self.matrices[
                np.clip(
                    np.searchsorted(times, query, side="right") - 1,
                    0,
                    len(keyframes) - 1,
                )
            ]
        else:
            # the keyframe on the right of each query time, clipped
            # so the bracketing pair is always in-bounds
            upper = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
            lower = upper - 1

            # duplicated keyframe times are a zero length interval which
            # would otherwise divide by zero here
            span = times[upper] - times[lower]
            usable = span > 0
            # clamp rather than extrapolate outside of the keyframes
            blend = np.clip(
                np.where(
                    usable, (query - times[lower]) / np.where(usable, span, 1.0), 0.0
                ),
                0.0,
                1.0,
            )
            fraction = blend.reshape((-1, 1))
            # gather each side once, every channel below wants them
            before, after = keyframes[lower], keyframes[upper]

            if self.interpolation == "cubic":
                width = span.reshape((-1, 1))
                translation, quaternion, scale = (
                    hermite(
                        before[f],
                        after[f],
                        before[f + "_out"],
                        after[f + "_in"],
                        fraction,
                        width,
                    )
                    for f in _FIELDS
                )
                # per the glTF spec a cubic blends the quaternion elementwise
                # and normalizes, rather than traveling the arc the way a
                # linear one does, and that blend isn't a unit quaternion
                norm = np.linalg.norm(quaternion, axis=1).reshape((-1, 1))
                quaternion = quaternion / np.where(norm > 1e-12, norm, 1.0)
            else:
                # translation and scale interpolate linearly, but a rotation
                # has to travel the arc between keyframes or it will skew and
                # change speed on the way there
                translation, scale = (
                    before[f] * (1.0 - fraction) + after[f] * fraction
                    for f in ("translation", "scale")
                )
                quaternion = quaternion_slerp(
                    before["quaternion"], after["quaternion"], blend
                )

            sampled = tqs_matrix(translation, quaternion, scale)

        return sampled[0] if single else sampled

    # note there is deliberately no `apply(scene, time)` here: writing the
    # edge is `Scene.animate`, which keeps the graph the single place a
    # pose lives rather than having two spellings which could disagree.
