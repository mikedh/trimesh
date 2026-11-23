try:
    from . import generic as g
except BaseException:
    import generic as g

from PIL import Image, ImageSequence

from trimesh.interfaces import ffmpeg
from trimesh.scene.lighting import EnvironmentLight
from trimesh.viewer.render import render_scene

np = g.np
tf = g.trimesh.transformations
# H.264 qp of the coarsest constant quality in PROFILES: the quantizer
# step is 0.625 * 2 ** (qp / 6) and rounding errs at most half a step
STEP = 0.625 * 2 ** (18 / 6)


def require_ffmpeg():
    if not ffmpeg.exists:
        if g.all_dependencies:
            raise ValueError("ffmpeg is required!")
        g.pytest.skip("ffmpeg is not installed")


def test_video():
    require_ffmpeg()
    random = np.random.default_rng(0)
    with g.TemporaryDirectory() as directory:
        path = g.os.path.join(directory, "video.mp4")
        # (width, height, fps, channels) including an odd size
        for width, height, fps, channels in [(256, 144, 24, 4), (161, 97, 30, 3)]:
            # a smooth colored gradient with noise, different every frame
            ramp = np.linspace([40, 200, 90, 255], [200, 60, 160, 255], width)
            base = ramp[None, :, :channels] + random.integers(-4, 5, (height, width, 1))
            frames = np.clip(base + np.arange(3)[:, None, None, None], 0, 255)
            frames = frames.astype(np.uint8)
            padded = (width + width % 2, height + height % 2)
            # the lossless reference through the same color conversion: its
            # error is the 4:2:0 color, so the rest is the codec's
            reference = g.os.path.join(directory, "reference.mkv")
            command = f"ffmpeg -v error -y -f rawvideo -s {padded[0]}x{padded[1]}"
            command += f" -pix_fmt {ffmpeg.INPUT_FORMATS[channels]} -i -"
            command += " -vf scale=out_color_matrix=bt709:out_range=tv -c:v ffv1"
            command += " -pix_fmt yuv420p -colorspace bt709 -color_range tv "
            pad = ((0, height % 2), (0, width % 2), (0, 0))
            source = np.pad(frames[0], pad, "edge").tobytes()
            g.subprocess.run([*command.split(), reference], input=source, check=True)
            truth = g.decode_video(reference, padded)[0]
            ffmpeg.write_video(frames, path, fps)
            entries = "stream=width,height,r_frame_rate,pix_fmt"
            command = (
                f"ffprobe -v error -select_streams v:0 -of json -show_entries {entries}"
            )
            probe = g.subprocess.run(
                [*command.split(), path], capture_output=True, check=True
            )
            stream = g.json.loads(probe.stdout)["streams"][0]
            every = g.decode_video(path, padded)
            name = f"{width}x{height}@{fps}"
            assert len(every) == len(frames), f"{name} frames"
            assert (stream["width"], stream["height"]) == padded, f"{name} size"
            assert stream["r_frame_rate"] == f"{fps}/1", f"{name} fps"
            assert stream["pix_fmt"] == "yuv420p", f"{name} pixel format"
            # the padded row and column repeat the last of the frame
            error = np.abs(every[0].astype(float) - truth).mean()
            assert error <= STEP / 2, f"{name} codec error"
        # a frame of another size is an error rather than a corrupt video
        with g.pytest.raises(ValueError, match="but video is"):
            ffmpeg.write_video([np.zeros((64, 64, 3)), np.zeros((64, 32, 3))], path)


def test_animations():
    """
    Every raw frame is a still of the scene posed at its time, or a turntable
    orbiting the world axis nearest camera up, and every encoded frame is in
    order and on time, with lights in the scene so stills light the same.
    """
    g.require_rendering()
    fps, resolution = 3, (64, 48)
    turntable = g.trimesh.creation.box(extents=[3, 1, 2]).scene()
    camera = tf.euler_matrix(0.5, 0.2, 0.1) @ tf.translation_matrix([0, 0, 6])
    turntable.camera_transform = camera
    up = np.abs(camera[:3, 1]).argmax()
    axis = np.eye(3)[up] * np.sign(camera[up, 1])
    scenes = {"box moves": g.dolly_scene("box"), "camera rides": g.dolly_scene("dolly")}
    scenes["turntable"] = turntable
    for name, scene in scenes.items():
        g.set_environment(scene, EnvironmentLight(name="sky"))
        raw = np.array(g.render_frames(scene, resolution, fps), dtype=np.int64)
        for index in range(len(raw)):
            posed = scene.copy()
            posed.animate(index / fps)
            if name == "turntable":
                angle = 2 * np.pi * index / len(raw)
                orbit = tf.rotation_matrix(angle, axis, scene.centroid)
                posed.camera_transform = orbit @ camera
            still = g.png(posed.render(resolution=resolution))
            g.assert_images(raw[index], still, f"{name} frame {index} is posed", 2)
        assert np.abs(raw[0] - raw[-1]).max() > 1, f"{name} moves"
        # GIF counts durations in hundredths of a second, so 3 fps can't be
        # 333 ms every frame: frames end within half a tick of when they should
        exact = np.arange(1, len(raw) + 1) * 1000.0 / fps
        for file_type, tick in {"gif": 10, "webp": 1}.items():
            data = render_scene(scene, resolution, file_type=file_type, fps=fps)
            image = Image.open(g.trimesh.util.wrap_as_stream(data))
            # WebP reads a frame's duration when it decodes it
            frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(image)]
            durations = [frame.info["duration"] for frame in frames]
            frames = np.array([np.array(frame) for frame in frames], dtype=np.int64)
            assert (np.abs(np.cumsum(durations) - exact) <= tick / 2).all(), name
            # every frame in order, each nearest its own raw frame
            distance = np.abs(frames[:, None] - raw).sum(axis=(2, 3, 4))
            assert distance.argmin(axis=1).tolist() == list(range(len(raw))), name
        if not ffmpeg.exists:
            # `test_video` raises where ffmpeg is required
            continue
        with g.TemporaryDirectory() as directory:
            path = g.os.path.join(directory, "video.mkv")
            scene.render(path, resolution=resolution, fps=fps)
            video = g.decode_video(path, resolution).astype(np.float64)
        assert len(video) == len(raw), f"{name} frame count"
        assert (np.abs(video - raw[..., :3]).mean(axis=(1, 2, 3)) < 4.0).all(), name
