try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.interfaces import ffmpeg

# H.264 qp of the coarsest constant quality in PROFILES: the quantizer
# step is 0.625 * 2 ** (qp / 6) and rounding errs at most half a step
STEP = 0.625 * 2 ** (18 / 6)

# (width, height, fps, channels, H.264 level times ten): an odd size and
# 4K, which is above 1080p, at the smallest level of table A-1 which fits
FRAMES = 3
CASES = [(256, 144, 24, 4, 41), (161, 97, 30, 3, 41), (3840, 2160, 30, 3, 51)]


def probe(path):
    """Width, height, frame rate, pixel format and level of the video stream."""
    entries = "stream=width,height,r_frame_rate,pix_fmt,level"
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


def test_video():
    if not ffmpeg.exists:
        if g.all_dependencies:
            raise ValueError("ffmpeg is required!")
        g.pytest.skip("ffmpeg is not installed")
    random = g.np.random.default_rng(0)
    # without hardware encoders, as on a typical machine
    software = {"h264_not_an_encoder": "", "libx264": ffmpeg.PROFILES["libx264"]}
    with g.TemporaryDirectory() as directory:
        for width, height, fps, channels, level in CASES:
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
                assert len(every) == len(frames), f"{name} frames"
                assert (stream["width"], stream["height"]) == padded, f"{name} size"
                assert stream["r_frame_rate"] == f"{fps}/1", f"{name} fps"
                assert stream["pix_fmt"] == "yuv420p", f"{name} pixel format"
                assert stream["level"] == level, f"{name} level"
                assert strict.returncode == 0 and not strict.stderr, (
                    f"{name} strict decode"
                )
                assert error <= STEP / 2, f"{name} codec error"
                assert profiles is None or video.encoder == "libx264", f"{name} fallback"
        # a frame of another size is an error rather than a corrupt video
        with g.pytest.raises(ValueError, match="but video is"):
            encode(path, [g.np.zeros((64, 64, 3)), g.np.zeros((64, 32, 3))], 30)
        # 8K at 240 fps is past the largest level of H.264
        with g.pytest.raises(ValueError, match="too large"):
            encode(path, [g.np.zeros((4320, 7680, 3), dtype=g.np.uint8)], 240)
        # none working says which ffmpeg, what it tried and what to do instead
        with g.pytest.MonkeyPatch.context() as patch:
            patch.setattr(ffmpeg, "PROFILES", {"h264_nope": "", "h264_nor_this": ""})
            with g.pytest.raises(ValueError) as none_work:
                encode(path, frames[:1], 30)
        words = [ffmpeg._ffmpeg_executable, "h264_nope", "h264_nor_this", "gif"]
        assert all(w in str(none_work.value) for w in words), "no encoder is explained"


def test_without_ffmpeg():
    from PIL import Image, ImageSequence

    from trimesh.viewer.render import render_scene

    # Pillow writes animations without ffmpeg; a video says what to do instead
    g.require_rendering()
    # a box sliding across the view so every frame is different
    scene, fps, resolution = g.dolly_scene("box", 2.0), 3, (96, 72)
    raw = g.np.array(g.render_frames(scene, resolution, fps), dtype=g.np.int64)
    # GIF counts durations in hundredths of a second, so 3 fps can't be
    # 333 ms every frame: frames end within half a tick of when they should
    exact = g.np.arange(1, len(raw) + 1) * 1000.0 / fps
    with g.pytest.MonkeyPatch.context() as patch:
        patch.setattr(ffmpeg, "exists", False)
        for file_type, tick in {"gif": 10, "webp": 1}.items():
            data = render_scene(scene, resolution, file_type=file_type, fps=fps)
            image = Image.open(g.trimesh.util.wrap_as_stream(data))
            frames, durations = [], []
            for frame in ImageSequence.Iterator(image):
                frames.append(frame.convert("RGBA"))
                durations.append(frame.info["duration"])
            assert (g.np.abs(g.np.cumsum(durations) - exact) <= tick / 2).all()
            # every frame in order, each nearest its own raw frame
            distance = g.np.abs(g.np.array(frames, dtype=g.np.int64)[:, None] - raw)
            order = distance.sum(axis=(2, 3, 4)).argmin(axis=1).tolist()
            assert order == list(range(len(raw))), order
        with g.pytest.raises(ImportError) as missing:
            render_scene(scene, resolution, file_type="mp4", fps=fps)
    assert all(w in str(missing.value) for w in ["ffmpeg", "webp", "gif"])


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_without_ffmpeg()
    test_video()
