try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.interfaces import ffmpeg

# H.264 qp of the coarsest constant quality in PROFILES: the quantizer
# step is 0.625 * 2 ** (qp / 6) and rounding errs at most half a step
STEP = 0.625 * 2 ** (18 / 6)

# (width, height, fps, channels): an odd size and 4K, which is above 1080p
FRAMES = 3
CASES = [(256, 144, 24, 4), (161, 97, 30, 3), (3840, 2160, 30, 3)]


def probe(path):
    """Width, height, frame rate and pixel format of the video stream."""
    entries = "stream=width,height,r_frame_rate,pix_fmt"
    command = f"ffprobe -v error -select_streams v:0 -of json -show_entries {entries}"
    out = g.subprocess.run([*command.split(), path], capture_output=True, check=True)
    return g.json.loads(out.stdout)["streams"][0]


def encode(path, frames, fps, profiles=None):
    """Write frames to a video, with only `profiles` to pick from if given."""
    with g.pytest.MonkeyPatch.context() as patch:
        if profiles:
            patch.setattr(ffmpeg, "PROFILES", profiles)
        with ffmpeg.VideoWriter(path, fps) as video:
            for frame in frames:
                video.write(frame)
    return video


def test_h264_level():
    # smallest level of H.264 table A-1 which fits, up to 8K at 120 fps
    levels = "4.1 4.2 5.1 5.2 6 6.1 6.2".split()
    sizes = [(640, 360, 30), (1920, 1080, 60), (3840, 2160, 30), (3840, 2160, 60)]
    sizes += [(7680, 4320, 30), (7680, 4320, 60), (7680, 4320, 120)]
    for size, level in zip(sizes, levels):
        assert ffmpeg.h264_level(*size) == level, size
    with g.pytest.raises(ValueError):
        ffmpeg.h264_level(7680, 4320, 240)


def test_video():
    if not ffmpeg.exists:
        if g.all_dependencies:
            raise ValueError("ffmpeg is required!")
        g.pytest.skip("ffmpeg is not installed")
    random = g.np.random.default_rng(0)
    # without hardware encoders, as on a typical machine
    software = {"h264_not_an_encoder": "", "libx264": ffmpeg.PROFILES["libx264"]}
    checks = {}
    with g.TemporaryDirectory() as directory:
        for width, height, fps, channels in CASES:
            # a smooth gradient with noise, a little different every frame
            ramp = g.np.linspace(40, 200, width)[None, :, None]
            base = ramp + random.integers(
                -4, 5, (height, width, channels), dtype=g.np.int8
            )
            frames = g.np.clip(base + g.np.arange(FRAMES)[:, None, None, None], 0, 255)
            frames = frames.astype(g.np.uint8)
            padded = (width + width % 2, height + height % 2)
            # the lossless reference through the same color conversion: its
            # error is the 4:2:0 color, so the rest is the codec's
            reference = g.os.path.join(directory, "reference.mkv")
            command = f"ffmpeg -v error -y -f rawvideo -s {padded[0]}x{padded[1]}"
            command += f" -pix_fmt {'rgba' if channels == 4 else 'rgb24'} -i -"
            command += " -vf scale=out_color_matrix=bt709:out_range=tv -c:v ffv1"
            command += " -pix_fmt yuv420p -colorspace bt709 -color_range tv "
            pad = ((0, height % 2), (0, width % 2), (0, 0))
            g.subprocess.run(
                [*command.split(), reference],
                input=g.np.pad(frames[0], pad, "edge").tobytes(),
                check=True,
            )
            truth = g.decode_video(reference, padded)[0]
            for profiles in [None, software]:
                name = f"{width}x{height}@{fps} software={profiles is not None}"
                path = g.os.path.join(directory, "video.mp4")
                video = encode(path, frames, fps, profiles)
                stream = probe(path)
                strict = g.subprocess.run(
                    ["ffmpeg", "-v", "error", "-xerror", "-i", path, "-f", "null", "-"],
                    capture_output=True,
                )
                every = g.decode_video(path, padded)
                decoded = every[0].astype(float)
                # the padded row and column repeat the last of the frame
                error = g.np.abs(decoded - truth).mean()
                checks.update(
                    {
                        f"{name} frames": len(every) == len(frames),
                        f"{name} size": (stream["width"], stream["height"]) == padded,
                        f"{name} fps": stream["r_frame_rate"] == f"{fps}/1",
                        f"{name} pixel format": stream["pix_fmt"] == "yuv420p",
                        f"{name} strict decode": strict.returncode == 0
                        and not strict.stderr,
                        f"{name} codec error": error <= STEP / 2,
                        f"{name} fallback": profiles is None
                        or video.encoder == "libx264",
                    }
                )
        # a frame of another size is an error rather than a corrupt video
        with g.pytest.raises(ValueError, match="but video is"):
            encode(path, [g.np.zeros((64, 64, 3)), g.np.zeros((64, 32, 3))], 30)
        # none working says which ffmpeg, what it tried and what to do instead
        with g.pytest.MonkeyPatch.context() as patch:
            patch.setattr(ffmpeg, "PROFILES", {"h264_nope": "", "h264_nor_this": ""})
            with g.pytest.raises(ValueError) as none_work:
                encode(path, frames[:1], 30)
        words = [ffmpeg._ffmpeg_executable, "h264_nope", "h264_nor_this", "gif"]
        checks["no encoder is explained"] = all(w in str(none_work.value) for w in words)
    for name, ok in checks.items():
        assert ok, name


def test_without_ffmpeg():
    from PIL import Image, ImageSequence

    from trimesh.viewer import render

    # Pillow writes animations without ffmpeg; a video says what to do instead
    solid = g.np.zeros((6, 32, 24, 4), dtype=g.np.uint8)
    solid[..., 0] = (g.np.arange(6) * 40)[:, None, None]
    solid[..., 3] = 255
    fps = 30
    # GIF counts durations in hundredths of a second, so 30 fps can't be
    # 33 ms every frame: frames end within half a tick of when they should
    exact = g.np.arange(1, len(solid) + 1) * 1000.0 / fps
    with g.pytest.MonkeyPatch.context() as patch:
        patch.setattr(ffmpeg, "exists", False)
        for file_type, tick in {"gif": 10, "webp": 1}.items():
            data = render.encode(solid, None, file_type, fps)
            image = Image.open(g.trimesh.util.wrap_as_stream(data))
            frames, durations = [], []
            for frame in ImageSequence.Iterator(image):
                frames.append(frame.convert("RGBA"))
                durations.append(frame.info["duration"])
            assert (g.np.abs(g.np.cumsum(durations) - exact) <= tick / 2).all()
            # every frame in order, each nearest its own solid color
            distance = g.np.abs(g.np.array(frames, dtype=g.np.int64)[:, None] - solid)
            assert distance.sum(axis=(2, 3, 4)).argmin(axis=1).tolist() == list(range(6))
        with g.pytest.raises(ImportError) as missing:
            render.encode(solid, None, "mp4", fps)
    assert all(w in str(missing.value) for w in ["ffmpeg", "webp", "gif"])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_h264_level()
    test_without_ffmpeg()
    test_video()
