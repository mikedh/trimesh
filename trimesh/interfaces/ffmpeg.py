"""
ffmpeg.py
------------

Write H.264 video by piping frames to the system `ffmpeg`.
"""

import shutil
import subprocess
import tempfile

import numpy as np

from ..typed import ArrayLike, Integer, Number
from .generic import HIDDEN_CONSOLE

_ffmpeg_executable = shutil.which("ffmpeg")
exists = _ffmpeg_executable is not None

# H.264 table A-1 levels: the most macroblocks in a frame and a second,
# then the most bits a second and bits of buffer of High profile. A video
# declares the smallest level which fits so hardware decoders accept it,
# and is held to the peak of 4.2 which players decode in real time.
H264_LEVELS = {
    "4.1": (8192, 245760, 62500000, 78125000),
    "4.2": (8704, 522240, 62500000, 78125000),
    "5.1": (36864, 983040, 300000000, 300000000),
    "5.2": (36864, 2073600, 300000000, 300000000),
    "6": (139264, 4177920, 300000000, 300000000),
    "6.1": (139264, 8355840, 600000000, 600000000),
    "6.2": (139264, 16711680, 1000000000, 1000000000),
}
PEAK_BITRATE, PEAK_BUFFER = H264_LEVELS["4.2"][2:]

# Encoder to constant quality arguments, fastest first, all keeping SSIM
# above 0.98 on renders. NVENC's B-frames aren't references as some decoders
# can't take that. VLC 3 + NVIDIA VDPAU shreds H.264 above 4 MP: VLC's bug.
PROFILES = {
    "h264_nvenc": "-preset p5 -tune hq -rc vbr -cq 14 -b:v 0 -bf 2 -b_ref_mode disabled",
    "h264_videotoolbox": "-q:v 65",
    "libx264": "-preset veryfast -crf 18",
}

VIDEO_TYPES = ("mp4", "mkv", "mov")  # files which hold H.264
INPUT_FORMATS = {3: "rgb24", 4: "rgba"}  # channels to ffmpeg pixel format
# (size, *arguments) to if a test frame encoded
_FFMPEG_PROFILE_WORKS = {}


def h264_level(width: Integer, height: Integer, fps: Number) -> str:
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
    Encode frames into a video with the system ffmpeg, where the first
    frame sets the size and picks the fastest encoder which works for it.
    4:2:0 color needs an even size, so an odd width or height is padded
    by repeating the last column or row.
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
        self.encoder = self.shape = self.process = None
        # ffmpeg logs to a file as an unread pipe fills and blocks it
        self.log = tempfile.TemporaryFile()

    def write(self, frame: ArrayLike) -> None:
        """
        Add a frame to the video, blocking until ffmpeg has read it.

        Parameters
        ------------
        frame : (height, width, 3 or 4) uint8
          RGB or RGBA pixels with the first row at the top.
        """
        frame = np.asarray(frame, dtype=np.uint8)
        height, width, channels = frame.shape
        if self.shape is None:
            self.shape = frame.shape
            even = (width + width % 2, height + height % 2)
            size, level = "{}x{}".format(*even), h264_level(*even, self.fps)
            for self.encoder, quality in PROFILES.items():
                if H264_LEVELS[level][2] > PEAK_BITRATE:
                    # NVENC's constant quality overshoots the peak on bursts
                    # of detail: target the peak with the quality as the finest
                    quality = quality.replace("-cq", "-qmin")
                    quality = quality.replace("-b:v 0", f"-b:v {PEAK_BITRATE}")
                # a keyframe every two seconds, and BT.709 4:2:0 color which
                # is converted by the filter as ffmpeg before 7.1 only tags it
                arguments = (
                    f"-c:v {self.encoder} {quality} -profile:v high -level:v {level} "
                    + f"-maxrate {PEAK_BITRATE} -bufsize {PEAK_BUFFER} "
                    + f"-g {max(round(2 * self.fps), 1)} "
                    + "-vf scale=out_color_matrix=bt709:out_range=tv -pix_fmt yuv420p "
                    + "-colorspace bt709 -color_range tv"
                ).split()
                key = (size, *arguments)
                try:
                    works = _FFMPEG_PROFILE_WORKS[key]
                except KeyError:
                    # hardware encoders need hardware, drivers and a minimum size
                    test = f"-v error -f lavfi -i testsrc=size={size} -frames:v 1".split()
                    command = [_ffmpeg_executable, *test, *arguments, "-f", "null", "-"]
                    try:
                        # one frame takes far under a second, so a probe
                        # still running after 30 is a hung driver
                        run = subprocess.run(
                            command, capture_output=True, timeout=30, **HIDDEN_CONSOLE
                        )
                        works = run.returncode == 0
                    except (OSError, subprocess.TimeoutExpired):
                        works = False
                    _FFMPEG_PROFILE_WORKS[key] = works
                if works:
                    break
            else:
                raise ValueError(
                    f"no H.264 encoder works for {size} in `{_ffmpeg_executable}`, "
                    + f"tried {', '.join(PROFILES)}: export a `webp`, `avif` or `gif`"
                )
            source = (
                f"-v error -y -f rawvideo -pix_fmt {INPUT_FORMATS[channels]} -s {size} "
                + f"-r {self.fps} -color_primaries bt709 -color_trc bt709 -i -"
            ).split()
            if self.path.lower().endswith((".mp4", ".mov")):
                arguments += ["-movflags", "+faststart"]
            command = [_ffmpeg_executable, *source, *arguments, self.path]
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=self.log, **HIDDEN_CONSOLE
            )
        elif frame.shape != self.shape:
            raise ValueError(f"frame is {frame.shape} but video is {self.shape}")
        padded = np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)), "edge")
        # if ffmpeg failed this raises, and `close` then says why
        self.process.stdin.write(memoryview(padded))

    def close(self) -> None:
        """
        Finish encoding and wait for ffmpeg to write the file.
        """
        process, self.process = self.process, None
        if process is None:
            return
        # closes stdin, ignoring a pipe ffmpeg already broke, and waits
        process.communicate()
        self.log.seek(0)
        if process.returncode != 0:
            message = self.log.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed with `{self.encoder}`:\n{message}")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()
