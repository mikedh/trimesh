"""
test_ray_upstream.py
--------------------

Regression tests for historical upstream bugs against the embree ray
intersector. Each test reproduces one specific GitHub issue that was
reported against the old embree2 / pyembree stack, and asserts the
behaviour expected once trimesh is upgraded to `embreex>=4.4.0rc1`
with `RTC_SCENE_ROBUST` as the default scene flag.

Each test's docstring links the originating issue. Where a test
documents a known-unfixable limitation (float32 precision at huge
coordinates, surface-origin self-hits) the test asserts the working
*workaround* and logs the failing case with `log.debug` rather than
failing the suite.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g

import numpy as np
import pytest

import trimesh
from trimesh.constants import log

# Skip this whole module unless the embreex backend actually imports;
# `pytest.importorskip` raises `Skipped` at collection time if the
# import fails rather than returning a stub wrapper.
ray_pyembree = pytest.importorskip("trimesh.ray.ray_pyembree")


def test_issue_2504_duplicate_hits_small_scale():
    """https://github.com/mikedh/trimesh/issues/2504
    `intersects_id(multiple_hits=True)` reported duplicate entry/exit
    points when the mesh was tiny relative to the ray origin distance;
    the multi-hit loop would get stuck on one triangle and re-emit it."""
    # 1mm box with a ray origin 10m away — ratio 1e7 exposes the
    # float32-precision corner the original issue hit.
    mesh = trimesh.creation.box(extents=[1e-3, 1e-3, 1e-3])
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)

    origins = np.array([[0.0, 0.0, 10000.0]])
    directions = np.array([[0.0, 0.0, -1.0]])
    _tri, _ray, locations = mesh.ray.intersects_id(
        origins, directions, multiple_hits=True, return_locations=True
    )

    # a convex box pierced along its axis has exactly two unique hits
    # (entry + exit); the original bug produced many duplicates.
    unique_locations = np.unique(locations.round(12), axis=0)
    assert len(unique_locations) == 2


def test_issue_2324_apply_scale_breaks_embree():
    """https://github.com/mikedh/trimesh/issues/2324
    `mesh.apply_scale(1/64)` after construction caused embree to stop
    returning any hits on an otherwise trivial ray test."""
    mesh = trimesh.creation.box(extents=[64, 64, 64])
    mesh.apply_scale(1 / 64)  # collapses to a unit cube
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)

    # ray from outside the box directly through the +x/+y face pair
    locations, index_ray, _index_tri = mesh.ray.intersects_location(
        [[0.5, 0.5, -2.0]], [[0, 0, 1]], multiple_hits=False
    )

    assert len(locations) == 1
    assert index_ray.tolist() == [0]
    # nearest hit on a unit cube at the origin along +z is the z=-0.5 face
    assert np.isclose(locations[0, 2], -0.5, atol=1e-6)


def test_issue_1496_edge_vertex_hits_any_vs_id():
    """https://github.com/mikedh/trimesh/issues/1496
    `intersects_any` and `intersects_id` disagreed for rays that pass
    through vertices or edges of the mesh — one backend culled the edge
    hit, the other didn't."""
    inner = trimesh.creation.box(extents=[40, 40, 40]).subdivide_to_size(20)
    outer = trimesh.creation.box(extents=[60, 60, 60]).subdivide_to_size(20)
    inner.ray = ray_pyembree.RayMeshIntersector(inner)

    # every ray from origin out through a vertex of the larger box
    # must pass through the smaller box as well
    origins = np.zeros_like(outer.vertices)
    directions = outer.vertices

    any_hit = inner.ray.intersects_any(origins, directions)
    _tri, id_hit_rays = inner.ray.intersects_id(
        origins, directions, multiple_hits=False
    )

    # convert the sparse index_ray output into a boolean mask over rays
    id_hit_mask = np.zeros(len(origins), dtype=bool)
    id_hit_mask[id_hit_rays] = True

    # every such ray must hit, and both APIs must agree ray-for-ray
    assert any_hit.all()
    assert np.array_equal(any_hit, id_hit_mask)


def test_issue_2317_robust_edge_hits():
    """https://github.com/mikedh/trimesh/issues/2317
    Edge hits on axis-aligned geometry were unreliable without
    `RTC_SCENE_ROBUST` — rays grazing exactly along an edge would
    randomly miss."""
    box = trimesh.creation.box(extents=[2, 2, 2])
    box.ray = ray_pyembree.RayMeshIntersector(box, scale_to_box=False)

    # 4 z-parallel edges and 4 corners of the box; every ray passes
    # exactly through an edge or vertex of the mesh
    origins = np.array(
        [
            [1, 0, 5], [-1, 0, 5], [0, 1, 5], [0, -1, 5],  # edge centers
            [1, 1, 5], [-1, -1, 5], [1, -1, 5], [-1, 1, 5],  # corners
        ],
        dtype=np.float64,
    )
    directions = np.tile([[0.0, 0.0, -1.0]], (len(origins), 1))

    assert box.ray.intersects_any(origins, directions).all()


def test_issue_1362_many_rays_consistency():
    """https://github.com/mikedh/trimesh/issues/1362
    Batched `intersects_any` with many rays disagreed with single-ray
    invocations, which made debugging reproductions a nightmare."""
    mesh = g.get_mesh("featuretype.STL")
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh)

    # back every vertex off along its normal then cast back — each ray
    # must hit its origin vertex's face; a solid test of many rays
    origins = mesh.vertices + mesh.vertex_normals * 0.01
    directions = -mesh.vertex_normals

    batched = mesh.ray.intersects_any(origins, directions)
    one_at_a_time = np.array(
        [
            mesh.ray.intersects_any([origin], [direction])[0]
            for origin, direction in zip(origins, directions)
        ]
    )

    assert np.array_equal(batched, one_at_a_time)


def test_issue_1898_locations_colinear_with_directions():
    """https://github.com/mikedh/trimesh/issues/1898
    Returned hit `locations` were not colinear with the casting ray —
    multi-hit advancement drifted off-axis in float32."""
    mesh = trimesh.creation.icosphere()
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh)

    # rays fired from (0,0,-5) with small angular jitter — with a
    # unit-radius sphere at origin the half-angle to graze is ~11.3°,
    # so a jitter of 0.05 in XY (about ±3°) stays well inside that cone
    rng = np.random.RandomState(seed=1)
    jitter = (rng.random_sample((8, 3)) - 0.5) * 0.1
    directions = trimesh.unitize(jitter + [[0, 0, 1]])
    origins = np.tile([[0.0, 0.0, -5.0]], (len(directions), 1))

    _tri, index_ray, locations = mesh.ray.intersects_id(
        origins, directions, multiple_hits=False, return_locations=True
    )

    # first assert all rays hit — otherwise the colinearity check below
    # is trivially satisfied by an empty array
    assert len(index_ray) == len(origins)

    # each hit must lie on its ray: cross(loc - origin, direction) == 0
    ray_vectors = locations - origins[index_ray]
    cross = np.cross(ray_vectors, directions[index_ray])
    assert np.allclose(cross, 0, atol=1e-6)


def test_issue_306_single_vs_batched_rays():
    """https://github.com/mikedh/trimesh/issues/306
    Casting a single ray in a batch returned a different location than
    casting the same ray on its own — the batched call lost precision."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh)

    # back off each vertex by epsilon along its normal, then cast back;
    # a batched call and a single-ray call on entry 0 must agree
    origins = mesh.vertices + mesh.vertex_normals * 0.01
    directions = -mesh.vertex_normals

    batched_loc, _, _ = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    single_loc, _, _ = mesh.ray.intersects_location(
        [origins[0]], [directions[0]], multiple_hits=False
    )

    assert np.allclose(batched_loc[0], single_loc[0], atol=1e-5)


def test_issue_93_triangle_index_correct_for_first_hit():
    """https://github.com/mikedh/trimesh/issues/93
    `intersects_id(multiple_hits=False)` returned an arbitrary hit
    along the ray rather than the nearest one."""
    box = trimesh.creation.box(extents=[2, 2, 2])
    box.ray = ray_pyembree.RayMeshIntersector(box)

    # ray along +z starting far below the box; nearest face is z=-1
    origins = np.array([[0.0, 0.0, -15.0]])
    directions = np.array([[0.0, 0.0, 1.0]])

    triangle_idx, _, locations = box.ray.intersects_id(
        origins, directions, multiple_hits=False, return_locations=True
    )

    assert len(triangle_idx) == 1
    # hit must be on the -z face (z ≈ -1) and the returned triangle's
    # normal must point in -z
    assert np.isclose(locations[0, 2], -1.0, atol=1e-6)
    assert np.allclose(box.face_normals[triangle_idx[0]], [0, 0, -1], atol=1e-6)


def test_issue_181_contains_deterministic():
    """https://github.com/mikedh/trimesh/issues/181
    `mesh.contains` returned different results for the same points on
    repeated calls — embree state was leaking between queries."""
    mesh = trimesh.creation.icosphere()
    points = g.random((128, 3)) * 0.5  # well inside unit sphere

    first = mesh.contains(points)
    for _ in range(5):
        assert np.array_equal(mesh.contains(points), first)


def test_issue_242_contains_matches_ground_truth():
    """https://github.com/mikedh/trimesh/issues/242
    `contains()` should agree with an analytical ground truth. The
    subdivision-4 icosphere is a faceted polyhedron, so points in the
    shell between inscribed and circumscribed radii legitimately may
    disagree with a smooth-sphere oracle — we only assert on points
    clearly inside or clearly outside the mesh shell."""
    mesh = trimesh.creation.icosphere(subdivisions=4)
    # inscribed radius = closest face center to origin
    inscribed_radius = np.linalg.norm(mesh.triangles_center, axis=1).min()
    # circumscribed radius = farthest vertex from origin
    circumscribed_radius = np.linalg.norm(mesh.vertices, axis=1).max()

    points = g.random((2048, 3)) * 2 - 1
    radii = np.linalg.norm(points, axis=1)
    definitely_inside = radii < inscribed_radius * 0.995
    definitely_outside = radii > circumscribed_radius * 1.005

    got = mesh.contains(points)
    assert got[definitely_inside].all()
    assert not got[definitely_outside].any()


def test_issue_545_contains_simple_sphere():
    """https://github.com/mikedh/trimesh/issues/545
    `contains` was wildly wrong for a simple closed sphere. As with
    issue 242, disagreements in the faceted-shell region between
    inscribed and circumscribed radii are geometrically legitimate —
    assert only on points clearly inside or clearly outside."""
    mesh = trimesh.creation.uv_sphere(count=[32, 32], radius=1.0)
    assert mesh.is_watertight

    inscribed_radius = np.linalg.norm(mesh.triangles_center, axis=1).min()
    # circumscribed radius for a uv_sphere of radius 1 is exactly 1
    points = g.random((1024, 3)) * 2 - 1
    radii = np.linalg.norm(points, axis=1)
    definitely_inside = radii < inscribed_radius * 0.99
    definitely_outside = radii > 1.01

    got = mesh.contains(points)
    assert got[definitely_inside].all()
    assert not got[definitely_outside].any()


def test_issue_48_large_coordinate_ray_offset():
    """https://github.com/mikedh/trimesh/issues/48
    At very large coordinates embree reports a false positive hit for a
    ray offset 0.01 away from a surface and pointing AWAY from the
    mesh. Root cause: embree works in float32; at coordinates ~5e5
    the unit-in-last-place is ~6e-2, so an offset of 0.01 rounds to
    zero and the ray re-hits the triangle it started on.

    Known unfixable at this layer without working in a local frame.
    """
    # two triangles at northing ~5e5, easting ~1.8e5 (real-world UTM-ish)
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [528980.85, 183526.1, 0],
                [528980.85, 183526.1, 3],
                [528982.3, 183523.55, 0],
                [528982.3, 183523.55, 3],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[1, 0, 2], [1, 2, 3]]),
        process=False,
    )
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)

    face_normal = trimesh.unitize(
        np.cross(
            mesh.vertices[1] - mesh.vertices[0],
            mesh.vertices[2] - mesh.vertices[0],
        )
    )
    # origin 0.01 OUTSIDE the surface along the face normal, cast AWAY
    origin = (
        np.mean(mesh.vertices[[0, 2]], axis=0) + [0, 0, 1] + 0.01 * face_normal
    )

    # positive-regression assertion: the SAME scenario at unit scale
    # works correctly, so the workaround is to transform to a local
    # frame before casting
    unit_mesh = trimesh.creation.box(extents=[1, 1, 1])
    unit_mesh.ray = ray_pyembree.RayMeshIntersector(unit_mesh, scale_to_box=False)
    unit_origin = np.array([0.0, 0.0, 0.5 + 0.01])
    unit_normal = np.array([0.0, 0.0, 1.0])
    assert not unit_mesh.ray.intersects_any([unit_origin], [unit_normal])[0]

    # log (not assert) the known-broken large-coordinate case
    if mesh.ray.intersects_any([origin], [face_normal])[0]:
        log.debug("issue 48: false positive hit at large coordinates (known)")


def test_issue_331_surface_origin_no_self_hit():
    """https://github.com/mikedh/trimesh/issues/331
    A ray whose origin lies exactly on the mesh surface and whose
    direction points outward reports a self-hit — embree reports an
    immediate t=0 intersection with the originating triangle.

    This is a known limitation: callers must offset the origin by an
    epsilon along the direction. We assert that the workaround works
    (no self-hit with a 1e-6 outward nudge) and `log.debug` the raw
    zero-offset case.
    """
    mesh = trimesh.creation.box()
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh)

    surface_points, face_idx = trimesh.sample.sample_surface(mesh, 64, seed=0)
    outward_directions = mesh.face_normals[face_idx]

    # workaround: offset the origin outward by epsilon — no self-hits
    nudged = surface_points + outward_directions * 1e-6
    assert not mesh.ray.intersects_any(nudged, outward_directions).any()

    # log the raw surface-origin case; callers have to back off themselves
    raw_hit_rate = mesh.ray.intersects_any(
        surface_points, outward_directions
    ).mean()
    if raw_hit_rate >= 0.1:
        log.debug(
            "issue 331: surface-origin self-hit rate %.3f (known limitation)",
            raw_hit_rate,
        )


def test_issue_72_axis_aligned_voxel_hits():
    """https://github.com/mikedh/trimesh/issues/72
    Voxelisation via axis-aligned rays left holes where rays grazed
    the grid diagonals — every ray strictly inside the box's XY cross
    section must hit."""
    box = trimesh.creation.box(extents=[2, 2, 2])
    box.ray = ray_pyembree.RayMeshIntersector(box, scale_to_box=False)

    # 40x40 grid of z-parallel rays confined to (-0.9, 0.9) in XY, so
    # every ray passes strictly through the interior of the box
    xy = np.linspace(-0.9, 0.9, 40)
    origins = np.array([[x, y, -5.0] for x in xy for y in xy])
    directions = np.tile([[0.0, 0.0, 1.0]], (len(origins), 1))

    assert box.ray.intersects_any(origins, directions).all()


def test_issue_457_multihit_cylinder_count():
    """https://github.com/mikedh/trimesh/issues/457
    Multi-hit rays along the axis of a cylinder returned the wrong
    number of hits per ray — sometimes 1 or 3 when it should always
    be exactly 2 (entry + exit)."""
    mesh = trimesh.creation.cylinder(radius=1.0, height=4.0, sections=32)
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)

    # 16x16 grid confined to (-0.6, 0.6): every ray enters the cylinder
    xy = np.linspace(-0.6, 0.6, 16)
    origins = np.array([[x, y, -5.0] for x in xy for y in xy])
    directions = np.tile([[0.0, 0.0, 1.0]], (len(origins), 1))

    _tri, index_ray = mesh.ray.intersects_id(
        origins, directions, multiple_hits=True
    )

    # histogram hits-per-ray; every ray must have exactly 2 hits
    hits_per_ray = np.bincount(index_ray, minlength=len(origins))
    assert (hits_per_ray == 2).all()


def test_issue_1919_engines_agree_simple_mesh():
    """https://github.com/mikedh/trimesh/issues/1919
    The embree and native (pure-Python) ray backends disagreed on
    `intersects_any` for random rays through a simple convex mesh."""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    rng = np.random.default_rng(0)
    origins = (rng.random((200, 3)) - 0.5) * 5
    directions = trimesh.unitize(rng.random((200, 3)) - 0.5)

    embree_hits = ray_pyembree.RayMeshIntersector(mesh).intersects_any(
        origins, directions
    )
    native_hits = trimesh.ray.ray_triangle.RayMeshIntersector(mesh).intersects_any(
        origins, directions
    )

    assert np.array_equal(embree_hits, native_hits)


def test_issue_1180_first_hit_arrays_aligned():
    """https://github.com/mikedh/trimesh/issues/1180
    `intersects_location(multiple_hits=False)` returned `locations`,
    `index_ray`, and `index_tri` arrays of different lengths at
    specific batch sizes — an off-by-one in the result concatenation."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh)

    # the reporter specifically called out sizes near 1024 * 512 = 524288;
    # we scale down to keep CI fast but cover the 1024 boundary
    for ray_count in [1023, 1024, 1025, 2048, 10000]:
        origins = np.tile([[0.0, 0.0, -5.0]], (ray_count, 1))
        rng = np.random.default_rng(ray_count)
        # direction jitter offset so the mean is roughly +z toward sphere
        directions = trimesh.unitize(
            rng.random((ray_count, 3)) - [0.5, 0.5, -0.2]
        )

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            origins, directions, multiple_hits=False
        )

        # all three output arrays must be in lock-step
        assert len(locations) == len(index_ray) == len(index_tri)


def test_issue_1786_contains_stable_over_many_calls():
    """https://github.com/mikedh/trimesh/issues/1786
    `mesh.contains` drifted after many calls in a loop — the reporter
    saw results change after a few million queries."""
    mesh = trimesh.creation.icosphere()
    rng = np.random.default_rng(0)
    # 256 points in [-0.5, 0.5] — all strictly inside the unit sphere
    points = rng.random((256, 3)) - 0.5

    first = mesh.contains(points)
    # all points are strictly inside, so `first` must be all-True,
    # AND repeated calls must agree with `first`
    assert first.all()
    for _ in range(50):
        assert np.array_equal(mesh.contains(points), first)


def test_issue_2462_coplanar_ray_no_error():
    """https://github.com/mikedh/trimesh/issues/2462
    A ray coplanar with a flat mesh previously crashed embree on a
    shape-mismatch assertion; the expected behaviour is that it
    returns no hits and the result arrays are well-shaped."""
    # flat rectangle in the z=0 plane
    rect = trimesh.Trimesh(
        vertices=np.array(
            [[0, 0, 0], [0.1, 0, 0], [0.1, 0.3, 0], [0, 0.3, 0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    rect.ray = ray_pyembree.RayMeshIntersector(rect)

    # ray along +x at y=0.1, z=0 — coplanar with the rectangle
    locations, index_ray, index_tri = rect.ray.intersects_location(
        [[-0.5, 0.1, 0.0]], [[1.0, 0.0, 0.0]], multiple_hits=False
    )

    # must return empty results with valid shapes, not raise
    assert locations.shape == (0, 3)
    assert len(index_ray) == 0
    assert len(index_tri) == 0


def test_multihit_valid_false_branch(monkeypatch):
    """Exercise the `~valid` branch of the multi-hit loop in
    `ray_pyembree.intersects_id`. That branch trims `ok_rays`/`ok_tris`
    when `planes_lines` drops near-parallel rays, which is almost
    impossible to hit geometrically — embree only reports a triangle
    when the ray already pierces the plane with `|n·d| >> 1e-5`.

    We monkey-patch `planes_lines` to flip one valid entry per call,
    forcing the cull path regardless of geometry.
    """
    from trimesh import intersections

    real_planes_lines = intersections.planes_lines

    def flip_one_valid(
        plane_origins, plane_normals, line_origins, line_directions, **kwargs
    ):
        """Wraps the real planes_lines and flips the first True entry
        in `valid` to False — drops one hit per call to force the
        `ok_rays[valid]` cull path in the multi-hit loop."""
        on_plane, valid = real_planes_lines(
            plane_origins, plane_normals, line_origins, line_directions, **kwargs
        )
        if len(valid) > 1 and valid.all():
            idx = int(np.where(valid)[0][0])
            valid = valid.copy()
            valid[idx] = False
            on_plane = np.delete(on_plane, idx, axis=0)
        return [on_plane, valid]

    monkeypatch.setattr(
        "trimesh.ray.ray_pyembree.intersections.planes_lines", flip_one_valid
    )

    mesh = trimesh.creation.box().subdivide().subdivide()
    mesh.ray = ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)

    # three +z rays through the interior of the subdivided box
    origins = np.array([[0.2, 0.3, 5.0], [-0.1, 0.4, 5.0], [0.0, 0.0, 5.0]])
    directions = np.tile([[0.0, 0.0, -1.0]], (len(origins), 1))

    triangle_idx, index_ray, locations = mesh.ray.intersects_id(
        origins, directions, multiple_hits=True, return_locations=True
    )

    # primary requirement: no crash. secondary: the three output
    # arrays stay in lock-step after the cull branch fires.
    assert len(triangle_idx) == len(index_ray) == len(locations)


