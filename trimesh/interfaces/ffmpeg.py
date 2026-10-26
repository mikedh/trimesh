"""
ffmpeg.py
------------

Write H.264 video by piping frames to the system `ffmpeg`.
"""

import shutil
import subprocess
import tempfile
from contextlib import suppress

import numpy as np

from ..typed import Number
from .generic import HIDDEN_CONSOLE

_ffmpeg_executable = shutil.which("ffmpeg")
exists = _ffmpeg_executable is not None

# H.264 table A-1 levels: the most macroblocks in a frame and a second,
# then the most bits a second and bits of buffer of High profile. A video
# declares the smallest level which fits so hardware decoders accept it.
H264_LEVELS = {
    "4.1": (8192, 245760, 62500000, 78125000),
    "4.2": (8704, 522240, 62500000, 78125000),
    "5.1": (36864, 983040, 300000000, 300000000),
    "5.2": (36864, 2073600, 300000000, 300000000),
    "6": (139264, 4177920, 300000000, 300000000),
    "6.1": (139264, 8355840, 600000000, 600000000),
    "6.2": (139264, 16711680, 1000000000, 1000000000),
}

# Every video is held to level 4.2's bitrate and buffer, which players
# decode in real time: 4K noise at constant quality reaches 300 Mbps.
PEAK_BITRATE, PEAK_BUFFER = H264_LEVELS["4.2"][2:]

# Encoder to its constant quality arguments, fastest first: the first
# which works here for a size is used. NVENC at 14 and x264 at 18 both
# kept SSIM above 0.98 on renders. NVENC's B-frames aren't references
# (some decoders can't handle that). VLC 3 with NVIDIA's VDPAU shreds the
# color of any H.264 above about 4 megapixels from any encoder: that is
# VLC's bug, the files are conformant.
PROFILES = {
    "h264_nvenc": "-preset p5 -tune hq -rc vbr -cq 14 -b:v 0 -bf 2 -b_ref_mode disabled",
    "h264_videotoolbox": "-q:v 65",
    "libx264": "-preset veryfast -crf 18",
}

VIDEO_TYPES = ("mp4", "mkv", "mov")  # files which hold H.264
INPUT_FORMATS = {3: "rgb24", 4: "rgba"}  # channels to ffmpeg pixel format
# (width, height, arguments) to if a test frame encoded
_FFMPEG_PROFILE_WORKS = {}


def h264_level(width: int, height: int, fps: Number) -> str:
    """
    The smallest H.264 level which fits a frame size and frame rate.

    Parameters
    ------------
    width, height : int
      Frame size in pixels.
    fps : float
      Frames per second.

    Returns
    ------------
    level : str
      A key of `H264_LEVELS`.
    """
    macroblocks = -(-width // 16) * -(-height // 16)
    for level, (frame, rate, *_) in H264_LEVELS.items():
        if macroblocks <= frame and macroblocks * fps <= rate:
            return level
    raise ValueError(f"{width}x{height} at {fps} fps is too large for H.264")


class VideoWriter:
    """
    Encode frames into a video file with the system ffmpeg, which reads
    each frame as it is written: `write` blocks while ffmpeg is busy.
    The size is taken from the first frame. 4:2:0 color needs an even
    size, so an odd width or height is padded by repeating the last
    column or row: the video is a pixel larger than the frames.
    """

    def __init__(self, path: str, fps: Number = 30.0):
        """
        Parameters
        ------------
        path : str
          Output file, whose extension picks the container: `.mp4`, `.mkv`...
        fps : float
          Frames per second.
        """
        if not exists:
            raise ImportError(
                "video needs `ffmpeg` on the PATH (apt, brew, conda, ffmpeg.org): "
                + "or export an animated `webp`, `avif` or `gif` with only Pillow"
            )
        self.path, self.fps = path, fps
        self.encoder = self.shape = self.process = self.log = None

    def start(self, shape: tuple) -> None:
        """
        Pick the fastest encoder which works for a frame shape and
        start ffmpeg reading raw frames from stdin.

        Parameters
        ------------
        shape : (3,) int
          Height, width, and channels of every frame.
        """
        if len(shape) != 3 or shape[2] not in INPUT_FORMATS:
            raise ValueError(f"frames must be (h, w, 3 or 4), not {shape}")
        height, width, channels = self.shape = shape
        width, height = width + width % 2, height + height % 2
        level = h264_level(width, height, self.fps)
        for encoder, quality in PROFILES.items():
            if H264_LEVELS[level][2] > PEAK_BITRATE:
                # NVENC's constant quality overshoots the peak on bursts of
                # detail: target the peak with the quality as the finest
                quality = quality.replace("-cq", "-qmin")
                quality = quality.replace("-b:v 0", f"-b:v {PEAK_BITRATE}")
            # a keyframe every two seconds, and BT.709 4:2:0 color which is
            # converted by the filter as ffmpeg before 7.1 only tags it
            arguments = (
                f"-c:v {encoder} {quality} -profile:v high -level:v {level} "
                + f"-maxrate {PEAK_BITRATE} -bufsize {PEAK_BUFFER} "
                + f"-g {max(round(2 * self.fps), 1)} "
                + "-vf scale=out_color_matrix=bt709:out_range=tv -pix_fmt yuv420p "
                + "-colorspace bt709 -color_range tv"
            ).split()
            key = (width, height, *arguments)
            try:
                works = _FFMPEG_PROFILE_WORKS[key]
            except KeyError:
                # hardware encoders need hardware, drivers and a minimum size
                test = f"-v error -f lavfi -i testsrc=size={width}x{height} -frames:v 1"
                probe = [_ffmpeg_executable, *test.split(), *arguments, "-f", "null", "-"]
                try:
                    run = subprocess.run(
                        probe, capture_output=True, timeout=30, **HIDDEN_CONSOLE
                    )
                    works = run.returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    works = False
                _FFMPEG_PROFILE_WORKS[key] = works
            if works:
                break
        else:
            raise ValueError(
                f"no H.264 encoder works for {width}x{height} in "
                + f"`{_ffmpeg_executable}`, which tried: {', '.join(PROFILES)}. "
                + "Export an animated `webp`, `avif` or `gif` instead"
            )
        self.encoder = encoder
        source = (
            f"-v error -y -f rawvideo -pix_fmt {INPUT_FORMATS[channels]} "
            + f"-s {width}x{height} -r {self.fps} "
            + "-color_primaries bt709 -color_trc bt709 -i -"
        )
        command = [_ffmpeg_executable, *source.split(), *arguments]
        if self.path.lower().endswith((".mp4", ".mov")):
            command += ["-movflags", "+faststart"]
        # log to a file: an unread pipe fills and blocks ffmpeg
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            [*command, self.path],
            stdin=subprocess.PIPE,
            stderr=self.log,
            **HIDDEN_CONSOLE,
        )

    def write(self, frame) -> None:
        """
        Add a frame to the video, blocking until ffmpeg has read it.

        Parameters
        ------------
        frame : (height, width, 3 or 4) uint8
          RGB or RGBA pixels with the first row at the top.
        """
        frame = np.asarray(frame, dtype=np.uint8)
        if self.shape is None:
            self.start(frame.shape)
        elif frame.shape != self.shape:
            raise ValueError(f"frame is {frame.shape} but video is {self.shape}")
        height, width = frame.shape[:2]
        padded = np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)), "edge")
        try:
            self.process.stdin.write(memoryview(padded))
        except OSError:
            self.close()
            raise

    def close(self) -> None:
        """
        Finish encoding and wait for ffmpeg to write the file.
        """
        if self.process is None:
            return
        with suppress(OSError):
            self.process.stdin.close()
        returncode = self.process.wait()
        self.log.seek(0)
        message = self.log.read().decode("utf-8", errors="replace")
        self.log.close()
        self.process = None
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed with `{self.encoder}`:\n{message}")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()
