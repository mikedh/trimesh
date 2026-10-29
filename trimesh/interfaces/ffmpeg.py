"""
ffmpeg.py
------------

Write H.264 video by piping frames to the system `ffmpeg`.
"""

import shutil
import subprocess
import tempfile

import numpy as np

from ..typed import ArrayLike, Number
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

VIDEO_TYPES = ("mp4", "mkv", "mov")  # files which hold H.264
INPUT_FORMATS = {3: "rgb24", 4: "rgba"}  # channels to ffmpeg pixel format
# (size, *arguments) to whether a test frame encoded
_FFMPEG_PROFILE_WORKS = {}


class VideoWriter:
    """
    Encode frames to H.264 with the fastest working encoder, padding odd sizes.
    """

    def __init__(self, path: str, fps: Number = 30.0):
        """
        Parameters
        ------------
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
        self.path, self.fps = path, fps
        self.encoder = self.shape = self.process = None
        # an unread log pipe would fill and block ffmpeg
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
            # smallest level that fits, counting 16x16 macroblocks
            size, blocks = "{}x{}".format(*even), -(-even[0] // 16) * -(-even[1] // 16)
            fits = [
                level
                for level, (per_frame, per_second) in H264_LEVELS.items()
                if blocks <= per_frame and blocks * self.fps <= per_second
            ]
            if len(fits) == 0:
                raise ValueError(f"{size} at {self.fps} fps is too large for H.264")
            level = fits[0]
            for self.encoder, quality in PROFILES.items():
                if float(level) > 4.2:
                    # NVENC constant quality overshoots: target the peak instead
                    quality = quality.replace("-cq", "-qmin")
                    quality = quality.replace("-b:v 0", f"-b:v {PEAK_BITRATE}")
                # keyframes every two seconds; the filter converts to BT.709
                # as ffmpeg before 7.1 only tags it
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
                    # hardware encoders need drivers and a minimum size
                    test = f"-v error -f lavfi -i testsrc=size={size} -frames:v 1".split()
                    command = [_ffmpeg_executable, *test, *arguments, "-f", "null", "-"]
                    try:
                        # a one frame probe running 30 seconds is a hung driver
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
        # raises if ffmpeg died, and `close` says why
        self.process.stdin.write(memoryview(padded))

    def close(self) -> None:
        """
        Finish encoding and wait for ffmpeg to write the file.
        """
        process, self.process = self.process, None
        if self.log.closed:
            return
        with self.log:
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
