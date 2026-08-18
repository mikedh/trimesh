"""
animation.py
--------------

Keyframed animation of a scene, stored as translation,
rotation, and scale sampled at increasing times.
"""

import numpy as np

from .. import caching
from ..transformations import quaternion_slerp, tqs_from_matrix, tqs_matrix
from ..typed import Floating, Hashable, Literal, NDArray1D, NDArray3D, float64

# how values between keyframes are computed, which map onto the
# GLTF sampler interpolation modes of LINEAR, STEP, and CUBICSPLINE
Interpolation = Literal["linear", "step", "cubic"]

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

# the dtype is packed with no padding which lets it be viewed as a flat
# (n, 31) float array, where these are the columns of each group
# interpolating all ten channels at once beats doing it per-field.
# these index the 31 columns: a time, then three groups of ten
_VALUE = slice(1, 11)
_TANGENT_IN = slice(11, 21)
_TANGENT_OUT = slice(21, 31)

# and these index the ten columns *within* any one of those groups
_TRANSLATION = slice(0, 3)
_QUATERNION = slice(3, 7)
_SCALE = slice(7, 10)


def keyframes_from_matrix(times, matrices=None) -> NDArray1D:
    """
    Build a keyframe array from times and homogeneous transforms.

    Parameters
    ------------
    times : (n,) float
      Keyframe times in seconds.
    matrices : (n, 4, 4) float or None
      The transform at each time, or None for an identity
      at every keyframe.

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
        interpolation: Interpolation = "linear",
    ):
        """
        Create an animation from either matrices or a keyframe array.

        Parameters
        ------------
        frame_to : hashable
          The scene graph node this drives.
        frame_from : hashable or None
          Which node the keyframes are relative to, usually the parent
          of `frame_to`. Left as None it is the scene's base frame,
          matching `SceneGraph.update`.
        times : (n,) float or None
          Keyframe times in seconds, increasing. Pass with `matrices`.
        matrices : (n, 4, 4) float or None
          The `frame_from -> frame_to` transform at each time. Note this
          is the transform across one edge, not `scene.graph[frame_to]`
          which is the transform from the base frame.
        keyframes : (n,) KEYFRAME or None
          Keyframes to use directly, instead of `times` and `matrices`.
        name : str
          Animations sharing a name export as one glTF animation.
        interpolation : str
          How to blend between keyframes.

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
        The keyframes driving this animation.

        Returns
        ----------
        keyframes : (n,) KEYFRAME
          Time, transform, and cubic tangents of each keyframe.
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
    def interpolation(self) -> Interpolation:
        """
        How values between keyframes are computed.

        Returns
        ----------
        interpolation : str
          One of "linear", "step", or "cubic".
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: Interpolation) -> None:
        cleaned = str(value).strip().lower()
        if cleaned not in ("linear", "step", "cubic"):
            raise ValueError(f"unsupported interpolation `{value}`!")
        self._interpolation = cleaned

    def __len__(self) -> int:
        return len(self.keyframes)

    @property
    def duration(self) -> float:
        """
        How long this animation runs for.

        Note this is the time of the final keyframe rather than the
        span between the first and last, as that is what a viewer
        loops on for an animation which doesn't start at zero.

        Returns
        ----------
        duration : float
          Length of the animation, in seconds.
        """
        return float(self.keyframes["time"][-1])

    @caching.cache_decorator
    def matrices(self) -> NDArray3D[float64]:
        """
        The local transform of each keyframe.

        Returns
        ----------
        matrices : (n, 4, 4) float
          Local parent -> node transform at each keyframe.
        """
        keyframes = self.keyframes
        return tqs_matrix(
            keyframes["translation"], keyframes["quaternion"], keyframes["scale"]
        )

    def at(self, time: Floating) -> NDArray3D[float64]:
        """
        Sample the transform across this edge at one or more times.

        Times outside of the keyframe range are clamped to the
        first or last keyframe rather than extrapolated.

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

        keyframes = self.keyframes
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

            # the packed dtype means this is a zero-copy view, which lets all
            # ten interpolated channels blend in one operation rather than
            # per-field. `np.asarray` first as viewing a tracked array would
            # yield a tracked array, which then hashes itself on use
            flat = np.asarray(keyframes).view(float64).reshape((len(keyframes), -1))
            value = flat[:, _VALUE]
            # gather each side once, both branches want them more than once
            before, after = value[lower], value[upper]

            if self.interpolation == "cubic":
                # tangents are a rate of change so they scale with the interval
                width = span.reshape((-1, 1))
                squared = fraction**2
                cubed = fraction**3

                # the cubic Hermite basis, which is exactly the bracketing
                # keyframes at a fraction of zero and one
                tqs = (
                    (2.0 * cubed - 3.0 * squared + 1.0) * before
                    + width
                    * (cubed - 2.0 * squared + fraction)
                    * flat[lower, _TANGENT_OUT]
                    + (-2.0 * cubed + 3.0 * squared) * after
                    + width * (cubed - squared) * flat[upper, _TANGENT_IN]
                )
                # per the glTF spec a cubic blends the quaternion elementwise
                # and normalizes, rather than traveling the arc the way a
                # linear one does, and that blend isn't a unit quaternion
                norm = np.linalg.norm(tqs[:, _QUATERNION], axis=1).reshape((-1, 1))
                tqs[:, _QUATERNION] /= np.where(norm > 1e-12, norm, 1.0)
            else:
                # translation and scale interpolate linearly, but a rotation
                # has to travel the arc between keyframes or it will skew and
                # change speed on the way there
                tqs = before * (1.0 - fraction) + after * fraction
                tqs[:, _QUATERNION] = quaternion_slerp(
                    before[:, _QUATERNION], after[:, _QUATERNION], blend
                )

            sampled = tqs_matrix(
                tqs[:, _TRANSLATION], tqs[:, _QUATERNION], tqs[:, _SCALE]
            )
        return sampled[0] if single else sampled

    def resample(self, times) -> "RigidAnimation":
        """
        Return this animation sampled onto a new time base.

        Note that a spline has no tangents left after being resampled,
        as they only describe the curve through the original keyframes.

        Parameters
        ------------
        times : (m,) float
          Keyframe times for the result, in seconds.

        Returns
        ----------
        resampled : RigidAnimation
          The same motion, keyframed at the passed times.
        """
        return RigidAnimation(
            frame_to=self.frame_to,
            frame_from=self.frame_from,
            keyframes=keyframes_from_matrix(times, self.at(times)),
            name=self.name,
            interpolation="step" if self.interpolation == "step" else "linear",
        )

    # note there is deliberately no `apply(scene, time)` here: writing the
    # edge is `Scene.animate`, which is the only path that remembers what
    # was there first so it can be put back. two spellings which wrote the
    # same edge with different bookkeeping would disagree about the pose a
    # scene returns to.
