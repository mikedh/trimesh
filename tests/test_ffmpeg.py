try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.interfaces import ffmpeg

# H.264 table A-1: level to the most macroblocks in a frame and a second
LEVELS = {
    41: (8192, 245760),
    42: (8704, 522240),
    51: (36864, 983040),
    52: (36864, 2073600),
    60: (139264, 4177920),
    61: (139264, 8355840),
    62: (139264, 16711680),
}
# and the bits a second and bits of buffer of High profile level 4.2,
# which every player decodes in real time at any size
PLAYABLE = (62.5e6, 78.125e6)


def write(path, frames, fps, profile=None):
    """
    Encode frames into a video at `path` with only `profile` to choose
    from, or the automatic pick, and return the closed writer.
    """
    with g.pytest.MonkeyPatch.context() as patch:
        if profile is not None:
            patch.setattr(ffmpeg, "PROFILES", (profile,))
        with ffmpeg.VideoWriter(path, fps) as video:
            for frame in frames:
                video.write(frame)
    return video


def profiles(width, height, fps):
    """Every profile which works on this machine for a size and frame rate."""
    return [p for p in ffmpeg.PROFILES if ffmpeg.profile_works(p, width, height, fps)]


def particle_burst(width, height, fps, random):
    """
    One second of a still gradient for a third of it and then 2x2 pixel
    particles in random colors over a tenth of the frame flying apart.
    """
    calm = g.np.linspace(30, 90, width).astype(g.np.uint8)[None, :, None]
    calm = g.np.broadcast_to(calm, (height, width, 3))
    count = width * height // 40
    start = random.uniform(0.0, 1.0, (count, 2)) * [width - 1, height - 1]
    velocity = random.normal(0.0, 40.0, (count, 2))
    colors = random.integers(0, 256, (count, 1, 3), dtype=g.np.uint8)
    square = g.np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for index in range(fps):
        frame = calm.copy()
        if index >= fps // 3:
            moved = (start + velocity * (index - fps // 3)) % [width - 1, height - 1]
            corners = moved.astype(int)[:, None, :] + square
            frame[corners[..., 1], corners[..., 0]] = colors
        yield frame


def playback(path, width, height, fps, prefix):
    """Named checks of a video as a strict hardware decoder sees it."""
    colors = "color_space,color_primaries,color_transfer"
    entries = f"stream=profile,level,codec_tag_string,{colors}:packet=size,flags"
    command = "ffprobe -v error -select_streams v:0 -of json -show_entries".split()
    run = g.subprocess.run([*command, entries, path], capture_output=True)
    probe = g.json.loads(run.stdout)
    stream, packets = probe["streams"][0], probe["packets"]
    frame_limit, rate_limit = LEVELS.get(stream["level"], (0, 0))
    bitrate, buffer = PLAYABLE
    macroblocks = ((width + 15) // 16) * ((height + 15) // 16)
    # the decoder's buffer fills at the level's bitrate and every frame
    # takes its bits out, which must never take more than is there
    fullness, underflow = buffer, False
    for packet in packets:
        fullness = min(fullness + bitrate / fps, buffer) - 8 * int(packet["size"])
        underflow = underflow or fullness < 0
    keyframes = g.np.flatnonzero(["K" in packet["flags"] for packet in packets])
    gaps = g.np.diff([*keyframes, len(packets)])
    with open(path, "rb") as f:
        data = f.read()
    decode = ["ffmpeg", "-v", "error", "-err_detect", "explode", "-i", path]
    strict = g.subprocess.run([*decode, "-f", "null", "-"], capture_output=True)
    checks = {
        "decodes strictly": strict.returncode == 0 and len(strict.stderr) == 0,
        "high profile": stream["profile"] == "High",
        # parameter sets in the sample description rather than the samples
        "avc1 sample entry": stream["codec_tag_string"] == "avc1",
        "level fits": macroblocks <= frame_limit and macroblocks * fps <= rate_limit,
        "bitrate plays in real time": not underflow,
        "tagged with its colors": [stream.get(key) for key in colors.split(",")]
        == ["bt709"] * 3,
        "keyframe every two seconds": gaps.max() <= 2 * fps,
        "index at the front": data.find(b"moov") < data.find(b"mdat"),
    }
    return {f"{prefix} {name}": ok for name, ok in checks.items()}


def test_without_ffmpeg():
    from PIL import Image, ImageSequence

    from trimesh.viewer import render

    # Pillow writes animations without ffmpeg, and asking for a video
    # without ffmpeg says how to get it or what to export instead
    solid = g.np.zeros((6, 32, 24, 4), dtype=g.np.uint8)
    solid[..., 0] = (g.np.arange(6) * 40)[:, None, None]
    solid[..., 3] = 255
    fps = 30
    # the milliseconds each format counts a frame's duration in: GIF's
    # are hundredths of a second, so 30 fps can't be 33 ms every frame
    ticks = {"gif": 10, "webp": 1, "avif": 1}
    # when each frame should end, which a whole number of ticks can be
    # within half a tick of without drifting
    exact = g.np.arange(1, len(solid) + 1) * 1000.0 / fps
    checks = {}
    # the smallest level of H.264 table A-1 which fits, up to 8K at 120 fps
    for width, height, rate in [
        (640, 360, 30),
        (1920, 1080, 60),
        (3840, 2160, 60),
        (7680, 4320, 30),
        (7680, 4320, 60),
        (7680, 4320, 120),
    ]:
        macroblocks = -(-width // 16) * -(-height // 16)
        fits = [
            level
            for level, (frame, second) in LEVELS.items()
            if macroblocks <= frame and macroblocks * rate <= second
        ]
        checks[f"{width}x{height} {rate}fps level"] = (
            ffmpeg.h264_level(width, height, rate) == f"{min(fits) / 10:g}"
        )
    checks["levels are table A-1"] = {
        round(float(level) * 10): tuple(values[:2])
        for level, values in ffmpeg.H264_LEVELS.items()
    } == LEVELS
    with g.pytest.MonkeyPatch.context() as patch:
        patch.setattr(ffmpeg, "exists", False)
        for file_type, tick in ticks.items():
            data = render.encode(solid, None, file_type, fps)
            image = Image.open(g.trimesh.util.wrap_as_stream(data))
            frames, durations = [], []
            for frame in ImageSequence.Iterator(image):
                frames.append(frame.convert("RGBA"))
                durations.append(frame.info["duration"])
            ends = g.np.cumsum(durations)
            checks[f"{file_type} frames end on time"] = (
                len(ends) == len(exact)
                and (g.np.abs(ends - exact) <= tick / 2).all()
                and all(duration % tick == 0 for duration in durations)
            )
            # every frame in order, each nearest its own solid color
            distance = g.np.abs(
                g.np.array(frames, dtype=g.np.int64)[:, None] - solid
            ).sum(axis=(2, 3, 4))
            checks[f"{file_type} frames"] = distance.argmin(axis=1).tolist() == list(
                range(len(solid))
            )
        with g.pytest.raises(ImportError) as missing:
            render.encode(solid, None, "mp4", fps)
    words = ["ffmpeg", "install", "webp", "gif"]
    checks["missing ffmpeg is explained"] = all(w in str(missing.value) for w in words)
    for name, ok in checks.items():
        assert ok, name


def test_video():
    if not ffmpeg.exists:
        if g.all_dependencies:
            raise ValueError("ffmpeg is required!")
        g.pytest.skip("ffmpeg is not installed")

    checks = {}
    random = g.np.random.default_rng(0)
    with g.TemporaryDirectory() as directory:
        mkv = g.os.path.join(directory, "video.mkv")
        mp4 = g.os.path.join(directory, "video.mp4")
        # every frame is a different solid color so order and count are
        # checkable, with every profile which works here and the automatic pick
        colors = random.integers(25, 225, (24, 3), dtype=g.np.uint8)
        for width, height, channels in [(256, 144, 4), (161, 97, 3)]:
            # odd sizes are padded by a pixel to be even
            padded = (width + width % 2, height + height % 2)
            frames = g.np.full((24, height, width, channels), 255, g.np.uint8)
            frames[..., :3] = colors[:, None, None, :]
            for profile in [None, *profiles(*padded, 24)]:
                video = write(mkv, frames, 24, profile)
                decoded = g.decode_video(mkv, padded)[:, :height, :width]
                name = f"{video.encoder} {width}x{height}x{channels}"
                checks[f"{name} frame count"] = len(decoded) == len(colors)
                # lossy, but a solid color should survive closely
                error = g.np.abs(decoded.mean(axis=(1, 2)) - colors).max()
                checks[f"{name} colors in order"] = error < 6.0

        # constant quality keeps the detail of gray noise sliding sideways, which
        # a starved bitrate blocks and 4:2:0 color quarters, over three seconds
        # which need a keyframe partway through
        gray = random.integers(0, 256, (360, 640, 1), dtype=g.np.uint8).repeat(3, 2)
        sliding = g.np.array([g.np.roll(gray, 16 * i, axis=1) for i in range(12)])
        for profile in profiles(640, 360, 4):
            encoder = profile.split()[0]
            write(mp4, sliding, 4, profile)
            decoded = g.decode_video(mp4, (640, 360)).astype(g.np.int64)
            checks[f"{encoder} keeps detail"] = g.np.abs(decoded - sliding).mean() < 6.0
            checks.update(playback(mp4, 640, 360, 4, f"{encoder} 4fps"))

        # every profile plays everywhere at a normal and a high frame rate and
        # larger than 1080p, where the level alone allows 300 Mbps, even from a
        # burst of particles after a calm start, far more data than a player
        # decodes, which NVENC's constant quality alone overshoots at 4K
        sizes = [(1920, 1080, 30), (1920, 1080, 60), (2560, 1600, 30), (3840, 2160, 30)]
        for width, height, fps in sizes:
            for profile in profiles(width, height, fps):
                encoder = profile.split()[0]
                frames = particle_burst(width, height, fps, g.np.random.default_rng(1))
                write(mp4, frames, fps, profile)
                prefix = f"{encoder} {width}x{height} {fps}fps"
                checks.update(playback(mp4, width, height, fps, prefix))

        # ffmpeg's log is read once it exits, so more of it than a pipe
        # holds, from a verbose log or a warning every frame, can't block
        arguments = ffmpeg.PLAYBACK_ARGUMENTS
        with g.pytest.MonkeyPatch.context() as patch:
            patch.setattr(ffmpeg, "PLAYBACK_ARGUMENTS", (*arguments, "-v", "debug"))
            write(mkv, g.np.zeros((300, 144, 256, 3), g.np.uint8), 30)
            # a profile which can't take the arguments is skipped when
            # choosing, rather than chosen and then failing on the real video
            patch.setattr(ffmpeg, "PLAYBACK_ARGUMENTS", (*arguments, "-bogus", "1"))
            checks["bad arguments rejected by the probe"] = not profiles(256, 256, 30)
        checks["verbose log never blocks"] = len(g.decode_video(mkv, (256, 144))) == 300
        with g.pytest.MonkeyPatch.context() as patch:
            patch.setattr(ffmpeg, "PROFILES", ("h264_not_an_encoder", "libx264"))
            fallback = write(mp4, sliding[:1], 30).encoder
            # and with none working says which ffmpeg, what it tried, and
            # what to export instead
            patch.setattr(ffmpeg, "PROFILES", ("h264_not_an_encoder", "h264_nor_this"))
            with g.pytest.raises(ValueError) as none_work:
                write(mp4, sliding[:1], 30)
        checks["falls back to the next encoder"] = fallback == "libx264"
        words = [ffmpeg._ffmpeg_executable, "h264_not_an_encoder", "h264_nor_this"]
        checks["no encoder is explained"] = all(
            word in str(none_work.value) for word in [*words, "gif", "webp"]
        )

        # a frame of the wrong size is an error rather than a corrupt video
        wrong = [g.np.zeros((64, 64, 4), g.np.uint8), g.np.zeros((64, 32, 4), g.np.uint8)]
        with g.pytest.raises(ValueError, match="but video is"):
            write(mp4, wrong, 30)
    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_without_ffmpeg()
    test_video()
