try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.interfaces import ffmpeg

# H.264 table A-1 for the High profile: level to the most macroblocks in
# a frame, macroblocks a second, bits a second and bits of buffer
LEVELS = {
    41: (8192, 245760, 62.5e6, 78.125e6),
    42: (8704, 522240, 62.5e6, 78.125e6),
    51: (36864, 983040, 300e6, 300e6),
    52: (36864, 2073600, 300e6, 300e6),
}


def write(path, frames, settings):
    """Encode frames into a video at `path` and return the closed writer."""
    with ffmpeg.VideoWriter(path, settings) as video:
        for frame in frames:
            video.write(frame)
    return video


def playback(path, width, height, fps, prefix):
    """Named checks of a video as a strict hardware decoder sees it."""
    colors = "color_space,color_primaries,color_transfer"
    entries = f"stream=profile,level,codec_tag_string,{colors}:packet=size,flags"
    command = "ffprobe -v error -select_streams v:0 -of json -show_entries".split()
    run = g.subprocess.run([*command, entries, path], capture_output=True)
    probe = g.json.loads(run.stdout)
    stream, packets = probe["streams"][0], probe["packets"]
    frame_limit, rate_limit, bitrate, buffer = LEVELS.get(stream["level"], (0, 0, 0, 0))
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
        "bitrate fits the level": not underflow,
        "tagged with its colors": [stream.get(key) for key in colors.split(",")]
        == ["bt709"] * 3,
        "keyframe every two seconds": gaps.max() <= 2 * fps,
        "index at the front": data.find(b"moov") < data.find(b"mdat"),
    }
    return {f"{prefix} {name}": ok for name, ok in checks.items()}


def test_without_ffmpeg():
    from PIL import Image

    from trimesh.viewer import render

    # Pillow writes animations without ffmpeg, and asking for a video
    # without ffmpeg says how to get it or what to export instead
    solid = g.np.zeros((6, 32, 24, 4), dtype=g.np.uint8)
    solid[..., 0] = (g.np.arange(6) * 40)[:, None, None]
    solid[..., 3] = 255
    settings = ffmpeg.VideoSettings(fps=10)
    checks = {}
    with g.pytest.MonkeyPatch.context() as patch:
        patch.setattr(ffmpeg, "exists", False)
        for file_type in ["gif", "webp", "avif"]:
            data = render.encode(solid, file_type, settings)
            image = Image.open(g.trimesh.util.wrap_as_stream(data))
            checks[f"{file_type} frames"] = image.n_frames == len(solid)
        with g.pytest.raises(ImportError) as missing:
            render.encode(solid, "mp4", settings)
    words = ["ffmpeg", "install", "webp", "gif"]
    checks["missing ffmpeg is explained"] = all(w in str(missing.value) for w in words)
    for name, ok in checks.items():
        assert ok, name


def test_video():
    if not ffmpeg.exists:
        if g.all_dependencies:
            raise ValueError("ffmpeg is required!")
        g.log.warning("skipping ffmpeg tests")
        return

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
            encoders = [None, *(p.split()[0] for p in ffmpeg.profiles(*padded))]
            for encoder in encoders:
                video = write(mkv, frames, ffmpeg.VideoSettings(fps=24, encoder=encoder))
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
        for profile in ffmpeg.profiles(640, 360):
            encoder = profile.split()[0]
            write(mp4, sliding, ffmpeg.VideoSettings(fps=4, encoder=encoder))
            decoded = g.decode_video(mp4, (640, 360)).astype(g.np.int64)
            checks[f"{encoder} keeps detail"] = g.np.abs(decoded - sliding).mean() < 6.0
            checks.update(playback(mp4, 640, 360, 4, f"{encoder} 4fps"))

        # every profile plays everywhere even from new 1080p noise every frame,
        # far more data than its level allows, at a normal and a high frame rate
        noise = random.integers(0, 256, (6, 1080, 1920, 1), dtype=g.np.uint8).repeat(3, 3)
        for fps in [30, 60]:
            for profile in ffmpeg.profiles(1920, 1080, ffmpeg.VideoSettings(fps=fps)):
                encoder = profile.split()[0]
                frames = (noise[index % len(noise)] for index in range(fps))
                write(mp4, frames, ffmpeg.VideoSettings(fps=fps, encoder=encoder))
                checks.update(playback(mp4, 1920, 1080, fps, f"{encoder} {fps}fps"))

        # a profile which can't take the arguments is skipped when choosing,
        # rather than chosen and then failing on the real video
        bogus = ffmpeg.VideoSettings(arguments=("-bogus", "1"))
        checks["bad arguments rejected by the probe"] = not ffmpeg.profiles(
            settings=bogus
        )
        with g.pytest.MonkeyPatch.context() as patch:
            patch.setattr(ffmpeg, "PROFILES", ("h264_not_an_encoder", "libx264"))
            fallback = write(mp4, sliding[:1], None).encoder
        checks["falls back to the next encoder"] = fallback == "libx264"

        # a frame of the wrong size is an error rather than a corrupt video
        wrong = [g.np.zeros((64, 64, 4), g.np.uint8), g.np.zeros((64, 32, 4), g.np.uint8)]
        with g.pytest.raises(ValueError, match="but video is"):
            write(mp4, wrong, None)
    for name, ok in checks.items():
        assert ok, name


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_without_ffmpeg()
    test_video()
