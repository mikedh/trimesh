"""
animation_cycloidal.py
------------------------

Animate a cycloidal drive from a CAD assembly and export it
as a glTF file with keyframed animation.

A cycloidal drive is a reducer. This one has three crankshafts
rather than a single central eccentric: each turns on its own
axis at motor speed, and their eccentric journals push a pair of
lobed discs around a ring of fixed rollers. A disc has one fewer
lobe than there are rollers, so every motor revolution walks it
backwards by exactly one lobe, and the reduction is the lobe
count. The discs and the crankshaft group as a whole then turn
together at that slow output speed.

Render the result with `examples/render_blender.py`.
"""

import numpy as np

import trimesh
from trimesh.scene.animation import RigidAnimation, keyframes_from_matrix
from trimesh.scene.cameras import look_at
from trimesh.scene.lighting import PointLight

# the drive axis of this particular assembly
AXIS = [0.0, 0.0, 1.0]

# the disc has 19 lobes and runs against 20 rollers, so the
# drive reduces by 19:1 and the output turns the other way.
# both are measurable from the model: count the `bushing_roller`
# instances, or take the dominant angular harmonic of a disc
RATIO = 19

# how the assembly moves, keyed by part name. the 3DXML loader names
# every instance `<part>#<path ids>` so the prefix identifies the part
# anything not listed here is part of the housing and stays put
INPUT = {"camshaft", "vxb-6800-2rs"}
DISCS = {"disc_cam_A", "disc_cam_B"}
OUTPUT = {"output", "60355K430", "92320A138_spacer"}

# the drive is a sealed gearbox so from outside almost nothing appears to
# move. drop the top cover and the outer spacer ring to see the mechanism.
# note `fixed_lower` is the part at the *top* of this assembly
COVER = {"fixed_lower", "92510A120_spacer"}

# the housing plates are a big flat slab which mirrors the whole scene
# back at the camera if it's as polished as the moving parts, so they get
# a duller finish: it reads as a machined base rather than a mirror
HOUSING = {"fixed_top", "fixed_lower"}

# a believable albedo for machined steel, which is a lot darker than the
# near-white these parts are drawn in
STEEL = (0.58, 0.59, 0.63, 1.0)


def animate(scene, revolutions=3.0, duration=6.0, fps=30):
    """
    Add a cycloidal drive animation to a scene loaded from `cycloidal.3DXML`.

    Parameters
    ------------
    scene : trimesh.Scene
      The cycloidal assembly, modified in-place.
    revolutions : float
      How many turns of the input camshaft to animate.
    duration : float
      Length of the animation, in seconds.
    fps : int
      Keyframes per second.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene, with `animations` populated.
    """
    times = np.linspace(0.0, duration, int(duration * fps) + 1)
    # the input angle at every keyframe
    theta = np.linspace(0.0, revolutions * 2.0 * np.pi, len(times))

    # (n, 4, 4) stack of the output frame, which crawls backwards at the
    # reduction ratio and carries the whole crankshaft group with it
    spin_output = trimesh.transformations.rotation_matrix(-theta / RATIO, AXIS)

    cos, sin = np.cos(theta), np.sin(theta)

    # where the three crankshafts sit, which is what the input parts spin
    # about: a crankshaft turns on its *own* axis rather than the drive's
    cranks = np.array(
        [
            scene.graph.get(frame_to=n, frame_from=scene.graph.transforms.parents[n])[0][
                :3, 3
            ]
            for n in scene.graph.nodes
            if str(n).split("#")[0] == "camshaft"
        ]
    )

    for node in scene.graph.nodes:
        part = str(node).split("#")[0]
        parent = scene.graph.transforms.parents.get(node)
        if parent is None:
            # the root has no edge to animate
            continue

        # an animation drives one edge, and these keyframes are the
        # transform across it rather than `scene.graph[node]` from the
        # base frame, so the whole edge has to be carried along
        local = scene.graph.get(frame_to=node, frame_from=parent)[0]

        if part in INPUT:
            # each crankshaft turns on its own axis at motor speed while
            # the three of them are carried around the drive axis by the
            # output, so this is two rotations about two different points.
            # the journal bearings ride the same crank, which is why they
            # spin about it rather than about themselves
            axis = cranks[np.argmin(np.linalg.norm(cranks[:, :2] - local[:2, 3], axis=1))]
            matrices = spin_output @ (
                trimesh.transformations.rotation_matrix(theta, AXIS, point=axis) @ local
            )
        elif part in OUTPUT:
            matrices = spin_output @ local
        elif part in DISCS:
            # the disc is pushed around by the eccentric at motor speed
            # while turning on its own moving center at the output speed
            center = local[:3, 3]
            matrices = (
                trimesh.transformations.rotation_matrix(
                    -theta / RATIO, AXIS, point=center
                )
                @ local
            )
            # the eccentric orbits the disc center about the drive axis
            matrices[:, 0, 3] += cos * center[0] - sin * center[1] - center[0]
            matrices[:, 1, 3] += sin * center[0] + cos * center[1] - center[1]
        else:
            # housing, ring rollers and fasteners don't move
            continue

        scene.animations.append(
            RigidAnimation(
                frame_to=node,
                frame_from=parent,
                times=times,
                matrices=matrices,
                name="drive",
            )
        )

    return scene


def swept_corners(scene, samples=24):
    """
    Bounding box corners of a scene over its whole animation.

    An animated subject sweeps out much more space than its rest pose,
    and a camera fit to the rest pose alone will crop the motion.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to sample, restored to its rest pose before returning.
    samples : int
      How many times to sample the animation at.

    Returns
    ----------
    corners : (n, 3) float
      Bounds corners at every sampled time.
    """
    corners = [trimesh.bounds.corners(scene.bounds)]

    for moment in np.linspace(0.0, scene.duration, samples):
        scene.animate(moment)
        corners.append(trimesh.bounds.corners(scene.bounds))
    # put every edge the sampling moved back the way it was found
    scene.animate(None)

    return np.vstack(corners)


def orbit(
    scene,
    sweep=32.0,
    zoom=0.86,
    duration=6.0,
    keyframes=5,
    pad=1.25,
    elevation=55.0,
    fov=40.0,
    aspect=16.0 / 9.0,
    name=None,
):
    """
    Animate the scene camera on a gentle arc which drifts closer.

    The camera is a node in the scene graph like anything else, so moving
    it is the same `RigidAnimation` on the same graph the mechanism uses,
    and it exports as a GLTF camera a renderer picks up.

    Note the camera deliberately travels much less than the mechanism
    does. A full revolution turns everything in frame including the parts
    which are bolted down, which reads as the whole drive spinning rather
    than as a camera move.

    Only a handful of keyframes are stored and a cubic spline is fitted
    through them, which is what makes the move ease in and out instead of
    starting and stopping at speed.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to add a camera and camera animation to, modified in-place.
    sweep : float
      Total degrees to travel around the subject.
    zoom : float
      Fraction of the starting distance to end at, below one to push in.
    duration : float
      Length of the move, in seconds.
    keyframes : int
      How many poses to fit the spline through.
    pad : float
      How much empty space to leave around the subject at the start.
    elevation : float
      Degrees to tilt the camera down from the horizon. Left level
      with the drive axis the move would spin the camera in place.
    fov : float
      Vertical field of view, in degrees.
    aspect : float
      Width over height of the image this will be rendered at.
    name : str or None
      Which animation to add the camera to, None to join whatever is
      already in the scene.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene, with a camera and its animation.
    """
    if name is None:
        # animations sharing a name export as one GLTF animation, and a
        # renderer plays one at a time: Blender drops every animation but
        # the first into a muted NLA track, so a camera which had its own
        # would simply never move. join the mechanism instead.
        name = scene.animations[0].name if len(scene.animations) > 0 else "orbit"

    # a renderer keeps the vertical field of view and widens the frame to
    # match its own aspect ratio, so a camera fit against a narrower one
    # leaves the subject sitting small in the middle of the image
    half = np.radians(fov) / 2.0
    scene.camera.fov = np.degrees([2.0 * np.arctan(np.tan(half) * aspect), 2.0 * half])

    corners = swept_corners(scene)
    center = corners.mean(axis=0)

    # a camera with no rotation sits straight up the drive axis looking
    # down it, which is exactly the axis the move turns about, so tilt
    # it off that axis first or it would just spin the frame
    tilt = trimesh.transformations.rotation_matrix(
        np.radians(90.0 - elevation), [1.0, 0.0, 0.0]
    )
    # `look_at` returns a camera-to-world transform, which is what an edge
    # from the base frame to the camera node already is
    base = look_at(corners, fov=scene.camera.fov, rotation=tilt, pad=pad)

    times = np.linspace(0.0, duration, keyframes)
    # centered on the framing `look_at` picked so the subject stays put
    angles = np.radians(np.linspace(-sweep / 2.0, sweep / 2.0, keyframes))

    # swing that one pose about the subject rather than the origin, as an
    # (n, 4, 4) stack: a stacked `rotation_matrix` between two translations
    poses = (
        trimesh.transformations.translation_matrix(center)
        @ trimesh.transformations.rotation_matrix(angles, AXIS)
        @ trimesh.transformations.translation_matrix(-center)
    ) @ base

    # slide each pose along its own line to the subject, which zooms
    # without touching the field of view or where it is pointed
    scales = np.linspace(1.0, zoom, keyframes).reshape((-1, 1))
    poses[:, :3, 3] = center + (poses[:, :3, 3] - center) * scales

    # bracket the subject with the clip planes. a renderer sizes its depth
    # buffer against these and may bin its lights by depth too, so leaving
    # them at a default thousands of times larger than the scene throws
    # away the precision that local lighting needs
    far = np.linalg.norm(poses[:, :3, 3] - center, axis=1).max()
    scene.camera.z_near = far * 0.01
    scene.camera.z_far = far * 4.0

    frames = keyframes_from_matrix(times, poses)
    # the slope through each keyframe, which is what a Catmull-Rom spline
    # uses, and zero at the ends so the move eases in and out. note these
    # are per unit time as the sampler scales them by the interval
    step = times[1] - times[0]
    for field in ("translation", "quaternion", "scale"):
        tangent = np.zeros_like(frames[field])
        tangent[1:-1] = (frames[field][2:] - frames[field][:-2]) / (2.0 * step)
        frames[f"{field}_in"] = tangent
        frames[f"{field}_out"] = tangent

    scene.animations.append(
        RigidAnimation(
            frame_to=scene.camera.name,
            keyframes=frames,
            interpolation="cubic",
            name=name,
        )
    )
    return scene


def light(scene, warm=(255, 244, 224, 255), cool=(198, 216, 255, 255)):
    """
    Add a three-point light rig scaled to the scene.

    Lights are graph nodes like the camera, and export through
    `KHR_lights_punctual` so a renderer doesn't have to invent any.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to light, modified in-place.
    warm : (4,) uint8
      Color of the key light.
    cool : (4,) uint8
      Color of the fill and rim lights.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene, with `lights` populated.
    """
    corners = swept_corners(scene)
    center = corners.mean(axis=0)
    # a distance which is well outside the subject at any orientation
    radius = np.linalg.norm(np.ptp(corners, axis=0)) / 2.0
    # out near the shell: a light close to the subject throws its shadow
    # across the backdrop magnified by the ratio of the two distances,
    # so keeping them comparable keeps the shadow near the subject size
    distance = radius * 12.0

    # `KHR_lights_punctual` is in photometric units: a point light is in
    # candela. a physically based renderer converts that into its own,
    # i.e. Blender treats 683 lumens as a watt, so this is the candela
    # which lands one watt per square metre on a subject `distance` away
    unit = 683.0 * distance**2

    # these are all point lights rather than the more usual distant suns
    # because `lightbox` puts a shell around the whole scene: a sun is
    # infinitely far away, which is to say outside it, and the shell
    # would simply cast a shadow over everything
    rig = [
        # a bright key, a soft fill from the other side to keep the
        # shadows open, and a rim behind to edge the silhouette
        (PointLight(name="key", color=warm, intensity=70.0 * unit), [0.55, -0.65, 3.0]),
        (PointLight(name="fill", color=cool, intensity=20.0 * unit), [-1.1, -0.35, 2.4]),
        (PointLight(name="rim", color=cool, intensity=24.0 * unit), [-0.45, 1.1, 2.6]),
    ]

    lights = []
    for source, offset in rig:
        direction = np.array(offset, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        # a light points down its own -Z, the same way a camera does
        scene.graph[source.name] = _aim(center + direction * distance, center)
        lights.append(source)

    scene.lights = lights
    return scene


def _aim(position, target, up=AXIS):
    """
    Build a transform placing something at `position` facing `target`.

    Parameters
    ------------
    position : (3,) float
      Where to put it.
    target : (3,) float
      What it should point at.
    up : (3,) float
      Which way is up.

    Returns
    ----------
    matrix : (4, 4) float
      Transform whose -Z axis points from `position` to `target`.
    """
    position = np.asanyarray(position, dtype=np.float64)
    # GLTF cameras and lights both look down their own -Z
    back = position - np.asanyarray(target, dtype=np.float64)
    back /= np.linalg.norm(back)

    right = np.cross(np.asanyarray(up, dtype=np.float64), back)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        # looking straight along `up` leaves the cross product undefined
        right = np.cross([1.0, 0.0, 0.0], back)
        norm = np.linalg.norm(right)
    right /= norm

    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = np.cross(back, right)
    matrix[:3, 2] = back
    matrix[:3, 3] = position
    return matrix


def rescale(scene, factor):
    """
    Scale a scene in place without disturbing its graph.

    `Scene.scaled`, which `convert_units` goes through, rebuilds the
    graph flat and bakes one instance's transform into the geometry. That
    is fine for a static export and ruinous here: the crank axes this
    animation is built around stop being where the nodes say they are.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to scale, modified in-place.
    factor : float
      What to multiply every length by.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene, scaled.
    """
    # scaling the vertices and the edge translations by the same factor
    # scales the composed world transform by it exactly once
    for geometry in scene.geometry.values():
        geometry.apply_scale(factor)

    edges = scene.graph.transforms.edge_data
    for edge in list(edges.keys()):
        matrix = edges[edge].get("matrix")
        if matrix is None:
            continue
        moved = np.array(matrix, dtype=np.float64).copy()
        moved[:3, 3] *= factor
        scene.graph.update(frame_from=edge[0], frame_to=edge[1], matrix=moved)

    return scene


def smooth(scene):
    """
    Recover smooth shading on parts the CAD tessellation flattened.

    A curved face is only smooth if its triangles share vertices. This
    assembly arrives with a normal baked onto every corner, which splits
    them apart, and `fixed_top` comes out 57% flat-shaded and visibly
    faceted. Merging without regard to those normals puts the surfaces
    back together and lets the vertex normals be averaged again.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene whose geometries are merged in-place.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene.
    """
    for geometry in scene.geometry.values():
        if not hasattr(geometry, "merge_vertices"):
            continue
        geometry.merge_vertices(merge_norm=True)
        # drop the stored normals so they are averaged from the faces
        geometry.vertex_normals = None

    return scene


def polish(scene, metallic=0.92, roughness=0.13, color=STEEL, names=None):
    """
    Give parts an oiled machined-metal material.

    The parts carry their CAD colors as vertex colors, which GLTF
    multiplies against the material's base color, so this mostly sets how
    the surface *behaves* and `color` scales what is already there.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to apply materials to, modified in-place.
    metallic : float
      How metallic the parts are.
    roughness : float
      How rough the parts are, lower being glossier.
    color : (4,) float
      Multiplied against the CAD colors. These parts are drawn nearly
      white which is brighter than any real metal, and on a light
      background a white part has nothing to separate it from the
      backdrop, so this pulls them down to a believable albedo.
    names : iterable or None
      Which geometries to apply this to, None for all of them.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene, with PBR materials on the named geometries.
    """
    if names is None:
        names = scene.geometry.keys()

    for name in names:
        geometry = scene.geometry.get(name)
        visual = getattr(geometry, "visual", None)

        if hasattr(visual, "to_pbr"):
            # a `ColorVisuals` has nowhere to put a material, so move the
            # colors onto a texture visual as the vertex attribute GLTF
            # stores them as
            colors = visual.vertex_colors.copy()
            geometry.visual = trimesh.visual.TextureVisuals(
                material=visual.to_pbr(metallic=metallic, roughness=roughness)
            )
            geometry.visual.vertex_attributes["color"] = colors
            visual = geometry.visual
        elif not hasattr(visual, "material"):
            continue

        visual.material.metallicFactor = float(metallic)
        visual.material.roughnessFactor = float(roughness)
        visual.material.baseColorFactor = np.array(color, dtype=np.float64)

    return scene


def lightbox(
    scene,
    radius=18.0,
    slant=38.0,
    color=(0.97, 0.98, 1.0, 1.0),
    floor=(0.55, 0.56, 0.62, 1.0),
    name="lightbox",
):
    """
    Put the subject inside a white dome.

    A lightbox is a physical object: a white shell with the lights inside
    it. Building it as geometry means the file carries its own studio, so
    every renderer sees the same thing rather than each one having to be
    told separately how to fake an environment. It is also what the
    polished parts reflect, which is most of what makes them look
    polished.

    Held far enough away the subject's shadow spreads across the shell
    until it stops reading as a shadow at all, which leaves the clean
    gradient a backdrop wants. The gradient is tilted off vertical so it
    sweeps across the frame rather than sitting in level bands.

    Call this after `orbit` and `light`, which both frame themselves
    against the scene bounds and would otherwise fit to the dome.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to enclose, modified in-place.
    radius : float
      Dome radius as a multiple of the subject's own.
    slant : float
      Degrees to tilt the gradient off the drive axis.
    color : (4,) float
      Linear RGBA at the bright end of the shell.
    floor : (4,) float
      Linear RGBA at the dim end of the shell.
    name : str
      Node and geometry name for the dome.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene, with the dome added.
    """
    corners = swept_corners(scene)
    center = corners.mean(axis=0)
    size = np.linalg.norm(np.ptp(corners, axis=0)) / 2.0

    dome = trimesh.creation.icosphere(subdivisions=4, radius=size * radius)
    # we are looking at this from the inside, and GLTF culls back faces
    # by default, so the normals have to point inwards
    dome.invert()
    dome.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="lightbox",
            baseColorFactor=np.array(color, dtype=np.float64),
            metallicFactor=0.0,
            # a lightbox is matte: a glossy one would mirror the subject
            roughnessFactor=1.0,
        )
    )

    # a shell which is white in every direction gives polished metal
    # nothing dark to reflect, and it comes out looking like pale
    # plastic. darkening one end is the black card a photographer slides
    # under a subject for exactly that reason, and GLTF multiplies these
    # vertex colors against the base color
    axis = np.dot(
        trimesh.transformations.rotation_matrix(np.radians(slant), [1.0, 0.0, 0.0])[
            :3, :3
        ],
        AXIS,
    )
    blend = (np.dot(dome.vertices, axis) / (size * radius) + 1.0) / 2.0
    gradient = np.array(floor) + np.subtract(color, floor) * blend.reshape((-1, 1))
    dome.visual.vertex_attributes["color"] = trimesh.visual.color.to_rgba(
        np.clip(gradient / np.max(color), 0.0, 1.0)
    )

    scene.add_geometry(
        dome,
        geom_name=name,
        node_name=name,
        transform=trimesh.transformations.translation_matrix(center),
    )

    if scene.has_camera:
        # the far plane was set to bracket the subject, so push it out to
        # take in the shell which now surrounds it. only just far enough:
        # an oversized depth range is what costs a renderer its precision
        away = np.linalg.norm(scene.graph[scene.camera.name][0][:3, 3] - center)
        scene.camera.z_far = float(away + size * radius * 1.1)

    return scene


def to_y_up(scene):
    """
    Rotate a scene from trimesh's Z-up into the Y-up GLTF specifies.

    Every GLTF consumer assumes Y-up, so a Z-up file arrives on its side.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene to rotate, modified in-place.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene, rotated.
    """
    matrix = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0])

    base = scene.graph.base_frame
    roots = set(scene.graph.transforms.children[base])
    scene.apply_transform(matrix)

    # left-multiplying a quaternion is linear, so it can be written as the
    # matrix which turns `q` into `rotation * q` and applied to a stack
    w, x, y, z = trimesh.transformations.quaternion_from_matrix(matrix)
    left = np.array(
        [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]], dtype=np.float64
    )

    # `apply_transform` only rewrites the base frame's own edges, so any
    # animation driving one of those edges holds keyframes which are now
    # stale and have to travel exactly the way their edge just did.
    # rotate the stored translation and rotation rather than recomposing
    # matrices and taking them apart again, which would throw away the
    # cubic tangents and leave a spline with nothing to follow
    for animation in scene.animations:
        if animation.frame_to not in roots or animation.frame_from not in (None, base):
            continue
        keyframes = animation.keyframes.copy()
        # the value and both tangent groups all transform the same way as
        # this is a pure rotation, i.e. it has no translation of its own
        for group in ("", "_in", "_out"):
            keyframes[f"translation{group}"] = (
                keyframes[f"translation{group}"] @ matrix[:3, :3].T
            )
            keyframes[f"quaternion{group}"] = keyframes[f"quaternion{group}"] @ left.T
        animation.keyframes = keyframes

    return scene


def cutaway(scene):
    """
    Remove the cover so the moving parts inside are visible.

    Parameters
    ------------
    scene : trimesh.Scene
      The cycloidal assembly, modified in-place.

    Returns
    ----------
    scene : trimesh.Scene
      The passed scene with the cover geometry removed.
    """
    # geometry is instanced so removing it drops every instance
    scene.delete_geometry([name for name in scene.geometry if name in COVER])
    return scene


if __name__ == "__main__":
    import os

    here = os.path.abspath(os.path.dirname(__file__))
    scene = trimesh.load(os.path.join(here, "..", "models", "cycloidal.3DXML"))

    # GLTF is specified in metres and this assembly is drawn in
    # millimetres, so without this every renderer sees a gearbox 155
    # metres across. that is not just a label: `KHR_lights_punctual` is
    # photometric, so a point light 300 metres away needs about a billion
    # candela to light anything, which is past what a renderer will carry
    # through a half-float buffer and comes out pure black
    rescale(scene, trimesh.units.unit_conversion(scene.units, "m"))

    cutaway(scene)
    animate(scene)
    print(f"animated {len(scene.animations)} of {len(scene.graph.nodes)} nodes")

    # everything a renderer needs goes in the file: materials, a camera
    # and its motion, and lights, all in the orientation GLTF expects
    smooth(scene)
    polish(scene)
    # the housing is a big flat slab, so leave it matte enough to read as
    # a base rather than a mirror pointed back at the camera, and darker
    # so it doesn't dissolve into a light backdrop
    polish(
        scene,
        metallic=0.30,
        roughness=0.55,
        color=(0.34, 0.35, 0.39, 1.0),
        names=HOUSING,
    )
    # the upper lobe disc sits directly on top of the lower one and in
    # the same finish they read as one part: darkening it lets the lower
    # disc's lobes show through as the pair counter-orbit
    polish(
        scene,
        metallic=0.85,
        roughness=0.22,
        color=(0.20, 0.21, 0.24, 1.0),
        names={"disc_cam_B"},
    )
    orbit(scene)
    light(scene)
    # last, as it is much bigger than the subject and everything above
    # frames itself against the scene bounds
    lightbox(scene)
    to_y_up(scene)

    path = os.path.join(here, "cycloidal.glb")
    with open(path, "wb") as f:
        f.write(scene.export(file_type="glb"))
    print(f"wrote {path}")

    # step the scene graph through the animation to preview it, which is
    # also what a viewer callback would do every frame
    def callback(scene):
        import time

        scene.animate(time.time() % scene.duration)

    scene.show(callback=callback)
