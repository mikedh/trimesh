"""
ffmpeg.py
------------

Write video by piping frames to the system `ffmpeg`, using the
fastest encoder it has which actually works on this machine.

```
with VideoWriter("animation.mp4") as video:
    for time in times:
        scene.animate(time)
        video.write(viewer.to_array().color)
```
"""

import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field

import numpy as np

from ..constants import log
from ..typed import Number

_ffmpeg_executable = shutil.which("ffmpeg")
exists = _ffmpeg_executable is not None

# The levels of H.264 table A-1 a video can be: the most macroblocks in a
# frame and a second, and the most bits a second and bits of buffer of
# the High profile, which are 1.25 times the table's. A decoder for a level
# plays any video which fits in it, so a video declares the smallest
# which fits.
H264_LEVELS = {
    "4.1": (8192, 245760, 62500000, 78125000),
    "4.2": (8704, 522240, 62500000, 78125000),
    "5.1": (36864, 983040, 300000000, 300000000),
    "5.2": (36864, 2073600, 300000000, 300000000),
}

# The most bits a second and bits of buffer of level 4.2, which every
# video is held to: decoding takes time per bit, and this is what every
# hardware decoder and software player decodes in real time. A level 5
# allows 300 Mbps, which 4K noise reaches at constant quality and which
# stutters in software players.
PEAK_BITRATE, PEAK_BUFFER = H264_LEVELS["4.2"][2:]

# Video is H.264 High profile, which every player and hardware decoder
# plays, encoded the fastest way which works on this machine: an encoder
# and the arguments which hold its quality constant, tried in turn with
# exactly these arguments. On renders compared to lossless frames NVENC
# at 14 and x264 at 18 both kept SSIM above 0.98 within the level's
# bitrate, and x264 `veryfast` encodes 1080p30 in real time on 4 cores.
# NVENC's B-frames aren't references, which some decoders can't handle,
# and its constant quality `-cq` treats `PEAK_BITRATE` as soft, which only
# overshoots it at sizes where the level allows more: `output_arguments`
# targets `PEAK_BITRATE` there, with the constant quality as the finest.
# H.264 above 1080p is conformant but the least tested corner of hardware
# decoders: VLC 3 decoding with NVIDIA's VDPAU shreds the color of any
# H.264 above about 4 megapixels from any encoder, where ffmpeg doesn't.
PROFILES = (
    "h264_nvenc -preset p5 -tune hq -rc vbr -cq 14 -b:v 0 -bf 2 -b_ref_mode disabled",
    "h264_videotoolbox -q:v 65",
    "libx264 -preset veryfast -crf 18",
)

# rendered frames are sRGB, whose primaries and transfer are BT.709's,
# which ffmpeg carries from the input through to the video's tags
INPUT_COLOR = ("-color_primaries", "bt709", "-color_trc", "bt709")

# 4:2:0 color converted with the BT.709 matrix and tagged with it, which
# is what players assume for HD video even when they ignore the tags: the
# filter is what converts it in ffmpeg before 7.1, where tags are only tags
PLAYBACK_ARGUMENTS = tuple(
    (
        "-vf scale=out_color_matrix=bt709:out_range=tv -pix_fmt yuv420p "
        + "-colorspace bt709 -color_range tv"
    ).split()
)

# files which hold H.264 video
VIDEO_TYPES = ("mp4", "mkv", "mov")

# frame channels to the ffmpeg name for those pixels
INPUT_FORMATS = {3: "rgb24", 4: "rgba"}

# the milliseconds an animated image format counts frame durations in,
# where GIF's are hundredths of a second and the rest whole milliseconds
DURATION_TICKS = {"gif": 10}

# (profile, width, height, arguments) to whether it encoded a test frame
_FFMPEG_PROFILE_WORKS = {}


@dataclass
class VideoSettings:
    """
    How to encode a video, where the defaults are already good, and
    `profiles(width, height, settings)` lists which encoders work here.
    """

    fps: Number = 30.0
    # an encoder from `PROFILES` like "libx264", or None for the fastest working
    encoder: str | None = None
    # any other ffmpeg output arguments
    arguments: tuple = field(default_factory=tuple)


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


def frame_durations(count: int, fps: Number, file_type: str) -> list[int]:
    """
    How long each frame of an animated image shows, in whole ticks of
    its format, so every frame ends within half a tick of when it should
    rather than a rounded duration drifting: 30 fps in a GIF is 30, 40
    and 30 milliseconds rather than 30 every frame, which plays 11% fast.

    Parameters
    ------------
    count : int
      Frames in the animation.
    fps : float
      Frames per second.
    file_type : str
      The format, i.e. "gif", whose ticks are in `DURATION_TICKS`.

    Returns
    ------------
    durations : list[int]
      Milliseconds for each frame.
    """
    tick = DURATION_TICKS.get(file_type, 1)
    ends = np.round(np.arange(count + 1) * 1000.0 / (fps * tick)).astype(np.int64)
    return (np.diff(ends) * tick).tolist()


def output_arguments(
    profile: str, width: int, height: int, settings: VideoSettings
) -> list[str]:
    """
    The ffmpeg output arguments for one of `PROFILES` at a size, which
    write H.264 that plays everywhere.

    Parameters
    ------------
    profile : str
      An encoder and its arguments, i.e. "libx264 -preset veryfast -crf 18".
    width, height : int
      Frame size in pixels, which is even.
    settings : VideoSettings
      The frame rate and any extra arguments.

    Returns
    ------------
    arguments : list[str]
      Everything between the input and the output file.
    """
    encoder, *quality = profile.split()
    level = h264_level(width, height, settings.fps)
    if "-cq" in quality and H264_LEVELS[level][2] > PEAK_BITRATE:
        # NVENC's constant quality overshoots the peak on bursts of detail
        # where the level allows more, which a target holds: measured on 4K
        # particles it holds the cap, and calm 4K is 40% smaller at the
        # same SSIM, as it is never finer than the constant quality
        options = dict(zip(quality[::2], quality[1::2]))
        options["-qmin"] = options.pop("-cq")
        options["-b:v"] = str(PEAK_BITRATE)
        quality = [word for option in options.items() for word in option]
    # constant quality up to what every player decodes in real time, with
    # a keyframe every two seconds to seek to and recover from
    words = (
        f"-c:v {encoder} {' '.join(quality)} -profile:v high -level:v {level} "
        + f"-maxrate {PEAK_BITRATE} -bufsize {PEAK_BUFFER} "
        + f"-g {max(round(2 * settings.fps), 1)}"
    )
    return [*words.split(), *PLAYBACK_ARGUMENTS, *settings.arguments]


def profile_works(profile: str, width: int, height: int, settings: VideoSettings) -> bool:
    """
    Check if a profile can encode a frame of a size on this machine
    with the arguments a video would use: hardware encoders need the
    hardware, drivers which take every argument, and a minimum size.

    Parameters
    ------------
    profile : str
      An encoder and its arguments from `PROFILES`.
    width, height : int
      Frame size in pixels, which is even.
    settings : VideoSettings
      How the video will be encoded.

    Returns
    ------------
    works : bool
      If the profile encoded a test frame.
    """
    arguments = output_arguments(profile, width, height, settings)
    key = (profile, width, height, tuple(arguments))
    try:
        return _FFMPEG_PROFILE_WORKS[key]
    except KeyError:
        pass
    source = f"-v error -f lavfi -i testsrc=size={width}x{height}:rate=1 -frames:v 1"
    command = [_ffmpeg_executable, *source.split(), *arguments, "-f", "null", "-"]
    try:
        works = subprocess.run(command, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        works = False
    _FFMPEG_PROFILE_WORKS[key] = works
    return works


def profiles(
    width: int = 256, height: int = 256, settings: VideoSettings | None = None
) -> list[str]:
    """
    Every profile which works on this machine, fastest first.

    Parameters
    ------------
    width, height : int
      Frame size in pixels, which is even.
    settings : VideoSettings or None
      How the video will be encoded, defaults to `VideoSettings()`.

    Returns
    ------------
    working : list[str]
      Profiles from `PROFILES`.
    """
    if settings is None:
        settings = VideoSettings()
    return [
        profile
        for profile in PROFILES
        if profile_works(profile, width=width, height=height, settings=settings)
    ]


class VideoWriter:
    """
    Encode frames into a video file with the system ffmpeg.

    Frames are written to ffmpeg from a background thread so the next
    frame can be rendered while the last one is being encoded. The size
    is taken from the first frame and every frame after has to match.
    """

    def __init__(self, path: str, settings: VideoSettings | None = None):
        """
        Parameters
        ------------
        path : str
          Output file, whose extension picks the container: `.mp4`, `.mkv`...
        settings : VideoSettings or None
          How to encode, defaults to `VideoSettings()`.
        """
        if not exists:
            raise ImportError(
                "writing video needs the `ffmpeg` program on the PATH: install it "
                + "with `apt`, `brew`, `conda` or from ffmpeg.org, or export an "
                + "animated `webp`, `avif` or `gif` which only need Pillow"
            )
        self.path = path
        if settings is None:
            settings = VideoSettings()
        self.settings = settings
        # one of `PROFILES` and its encoder, picked when the first frame
        # gives the size unless an encoder was named
        self.profile = None
        self.encoder = settings.encoder
        self.shape = None
        self.process = None
        self.thread = None
        self.log = None
        # a few frames of slack so rendering and encoding overlap
        self.frames = queue.Queue(maxsize=4)
        self.error = None
        self.frame_count = 0

    def command(self, width: int, height: int, channels: int) -> list[str]:
        """
        The ffmpeg command reading raw frames of this size from stdin.

        Parameters
        ------------
        width, height : int
          Frame size in pixels.
        channels : int
          3 for RGB or 4 for RGBA.

        Returns
        ------------
        command : list[str]
          Arguments to run ffmpeg with.
        """
        settings = self.settings
        container = []
        if self.path.lower().endswith((".mp4", ".mov")):
            # the index at the front so a player starts without reading it all
            container = ["-movflags", "+faststart"]
        # raw frames in on stdin
        source = (
            f"-v error -y -f rawvideo -pix_fmt {INPUT_FORMATS[channels]} "
            + f"-s {width}x{height} -r {settings.fps}"
        )
        return [
            _ffmpeg_executable,
            *source.split(),
            *INPUT_COLOR,
            *("-i", "-"),
            *output_arguments(self.profile, width, height, settings),
            *container,
            self.path,
        ]

    def write(self, frame) -> None:
        """
        Add a frame to the video.

        Parameters
        ------------
        frame : (height, width, 3 or 4) uint8
          RGB or RGBA pixels with the first row at the top.
        """
        if self.error is not None:
            raise self.error
        frame = np.asarray(frame, dtype=np.uint8)
        if self.shape is None:
            if frame.ndim != 3 or frame.shape[2] not in INPUT_FORMATS:
                raise ValueError(f"frames must be (h, w, 3 or 4), not {frame.shape}")
            self.start(frame.shape)
        elif frame.shape != self.shape:
            raise ValueError(f"frame is {frame.shape} but video is {self.shape}")
        # 4:2:0 color needs an even size so an odd last row or column is
        # repeated, which also copies the frame into one contiguous block
        height, width = frame.shape[:2]
        self.frames.put(np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)), "edge"))
        self.frame_count += 1

    def start(self, shape: tuple) -> None:
        """
        Start ffmpeg and the thread feeding it for frames of a shape.

        Parameters
        ------------
        shape : (3,) int
          Height, width, and channels of every frame.
        """
        self.shape = shape
        height, width, channels = shape
        # frames are padded to an even size
        width, height = width + width % 2, height + height % 2
        settings = self.settings
        # a named encoder uses its profile, or is used as it is
        named = {profile.split()[0]: profile for profile in PROFILES}
        self.profile = named.get(self.encoder, self.encoder)
        if self.profile is None:
            # the fastest profile which works for this size on this machine,
            # trying each in turn so the ones after it are never run
            self.profile = next(
                (
                    profile
                    for profile in PROFILES
                    if profile_works(profile, width, height, settings)
                ),
                None,
            )
            if self.profile is None:
                tried = ", ".join(profile.split()[0] for profile in PROFILES)
                raise ValueError(
                    f"no H.264 encoder works for {width}x{height} in "
                    + f"`{_ffmpeg_executable}`, which tried: {tried}. Export "
                    + "an animated `webp`, `avif` or `gif` instead, which "
                    + "only need Pillow"
                )
        self.encoder = self.profile.split()[0]
        command = self.command(width=width, height=height, channels=channels)
        log.debug(f"running `{' '.join(command)}`")
        # ffmpeg's log goes to a file read once it exits: a pipe nobody
        # reads until then fills and blocks ffmpeg, and so every `write`
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self.log)
        self.thread = threading.Thread(target=self.feed, daemon=True)
        self.thread.start()

    def feed(self) -> None:
        """
        Write queued frames to ffmpeg until a None arrives. After an error
        frames are still taken off the queue, so `close` can't block on it.
        """
        while (frame := self.frames.get()) is not None:
            if self.error is not None:
                continue
            try:
                self.process.stdin.write(memoryview(frame))
            except OSError as E:
                # ffmpeg exited: surfaced by the next `write` or `close`
                self.error = E
        try:
            self.process.stdin.close()
        except OSError:
            pass

    def close(self) -> None:
        """
        Finish encoding and wait for ffmpeg to write the file.
        """
        if self.process is None:
            return
        self.frames.put(None)
        self.thread.join()
        returncode = self.process.wait()
        self.log.seek(0)
        message = self.log.read().decode("utf-8", errors="replace")
        self.log.close()
        self.process = None
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed with `{self.encoder}`:\n{message}")
        if self.error is not None:
            raise self.error

    def __enter__(self):
        """
        Use the writer as a context manager which closes it on exit.

        Returns
        ------------
        writer : VideoWriter
          This writer.
        """
        return self

    def __exit__(self, *args) -> None:
        """
        Finish writing the video.

        Parameters
        ------------
        args : tuple
          Exception information, which is not suppressed.
        """
        self.close()
