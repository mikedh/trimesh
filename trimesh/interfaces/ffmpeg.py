"""
ffmpeg.py
------------

Write H.264 video by piping frames to the system `ffmpeg`.
"""

import shutil
import subprocess
import tempfile
from itertools import chain

import numpy as np

from ..typed import ArrayLike, Integer, Iterable, Number
from .generic import HIDDEN_CONSOLE

_ffmpeg_executable = shutil.which("ffmpeg")
exists = _ffmpeg_executable is not None

# H.264 table A-1: max macroblocks per frame and per second for each level,
# with bitrate and buffer capped at High 4.2, which players decode in real time
H264_LEVELS = {
    "4.1": (8192, 245760),
    "4.2": (8704, 522240),
    "5.1": (36864, 983040),
    "5.2": (36864, 2073600),
    "6": (139264, 4177920),
    "6.1": (139264, 8355840),
    "6.2": (139264, 16711680),
}
PEAK_BITRATE, PEAK_BUFFER = 62500000, 78125000

# constant quality per encoder, fastest first, SSIM > 0.98; NVENC B-frames aren't
# references for some decoders. VLC 3 + VDPAU shreds H.264 above 4 MP: VLC's bug.
PROFILES = {
    "h264_nvenc": "-preset p5 -tune hq -rc vbr -cq 14 -b:v 0 -bf 2 -b_ref_mode disabled",
    "h264_videotoolbox": "-q:v 65",
    "libx264": "-preset veryfast -crf 18",
}
# above level 4.2 NVENC constant quality overshoots the peak, so target it
PEAK_PROFILES = {
    "h264_nvenc": f"-preset p5 -tune hq -rc vbr -qmin 14 -b:v {PEAK_BITRATE} -bf 2 "
    + "-b_ref_mode disabled"
}

VIDEO_TYPES = ("mp4", "mkv", "mov")  # files which hold H.264
INPUT_FORMATS = {3: "rgb24", 4: "rgba"}  # channels to ffmpeg pixel format
# {(width, height, fps) : output arguments of the first working encoder}
_FFMPEG_ARGUMENTS = {}


def encoder_arguments(width: Integer, height: Integer, fps: Number) -> list[str]:
    """
    H.264 output arguments for the fastest encoder that works at a size.

    Parameters
    ------------
    width, height : int
      Even frame size in pixels.
    fps : float
      Frames per second.

    Returns
    ------------
    arguments : list of str
      Codec, quality, level and color arguments for ffmpeg.
    """
    key = (width, height, fps)
    try:
        return _FFMPEG_ARGUMENTS[key]
    except KeyError:
        pass
    # smallest level that fits, counting 16x16 macroblocks
    blocks = -(-width // 16) * -(-height // 16)
    fits = [
        level
        for level, (per_frame, per_second) in H264_LEVELS.items()
        if blocks <= per_frame and blocks * fps <= per_second
    ]
    if len(fits) == 0:
        raise ValueError(f"{width}x{height} at {fps} fps is too large for H.264")
    level = fits[0]
    # a one frame test of every encoder, as hardware ones need drivers
    test = f"-v error -f lavfi -i testsrc=size={width}x{height} -frames:v 1".split()
    for encoder, quality in PROFILES.items():
        if float(level) > 4.2:
            quality = PEAK_PROFILES.get(encoder, quality)
        # keyframes every two seconds; the filter converts to BT.709
        # as ffmpeg before 7.1 only tags it
        arguments = (
            f"-c:v {encoder} {quality} -profile:v high "
            + f"-level:v {level} -maxrate {PEAK_BITRATE} -bufsize {PEAK_BUFFER} "
            + f"-g {max(round(2 * fps), 1)} "
            + "-vf scale=out_color_matrix=bt709:out_range=tv -pix_fmt yuv420p "
            + "-colorspace bt709 -color_range tv"
        ).split()
        command = [_ffmpeg_executable, *test, *arguments, "-f", "null", "-"]
        try:
            # a one frame probe running 30 seconds is a hung driver
            run = subprocess.run(
                command, capture_output=True, timeout=30, **HIDDEN_CONSOLE
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if run.returncode == 0:
            _FFMPEG_ARGUMENTS[key] = arguments
            return arguments
    raise ValueError(
        f"no H.264 encoder works for {width}x{height} in `{_ffmpeg_executable}`, "
        + f"tried {', '.join(PROFILES)}: export a `webp`, `avif` or `gif`"
    )


def write_video(frames: Iterable[ArrayLike], path: str, fps: Number = 30.0) -> None:
    """
    Encode frames to an H.264 file, padding odd sizes.

    Parameters
    ------------
    frames : iterable of (height, width, 3 or 4) uint8
      RGB or RGBA pixels with the first row at the top, all one size.
    path : str
      Output file, whose extension picks the container.
    fps : float
      Frames per second.
    """
    if not exists:
        raise ImportError(
            "video needs `ffmpeg` on the PATH (apt, brew, conda, ffmpeg.org): "
            + "or export an animated `webp`, `avif` or `gif` with only Pillow"
        )
    frames = iter(frames)
    first = np.asarray(next(frames), dtype=np.uint8)
    height, width, channels = first.shape
    # H.264 needs an even size, padded by repeating the last row and column
    pad = ((0, height % 2), (0, width % 2), (0, 0))
    size = (width + width % 2, height + height % 2)
    arguments = encoder_arguments(*size, fps)
    source = (
        f"-v error -y -f rawvideo -pix_fmt {INPUT_FORMATS[channels]} "
        + f"-s {size[0]}x{size[1]} -r {fps} -color_primaries bt709 "
        + "-color_trc bt709 -i -"
    ).split()
    if path.lower().endswith((".mp4", ".mov")):
        arguments = [*arguments, "-movflags", "+faststart"]
    # an unread log pipe would fill and block ffmpeg
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            [_ffmpeg_executable, *source, *arguments, path],
            stdin=subprocess.PIPE,
            stderr=log,
            **HIDDEN_CONSOLE,
        )
        try:
            for frame in chain([first], frames):
                frame = np.asarray(frame, dtype=np.uint8)
                if frame.shape != first.shape:
                    raise ValueError(f"frame is {frame.shape} but video is {first.shape}")
                process.stdin.write(memoryview(np.pad(frame, pad, "edge")))
        except BrokenPipeError:
            # ffmpeg died, and its log says why
            pass
        finally:
            process.communicate()
        if process.returncode != 0:
            log.seek(0)
            message = log.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed with `{arguments[1]}`:\n{message}")
