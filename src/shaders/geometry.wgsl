// ===========================================================================
// geometry.wgsl  –  Sphere intersection + scene traversal
//
// Depends on: common.wgsl  (Ray, Hit, SphereGpu, point_at, spheres binding)
// ===========================================================================

// ---------------------------------------------------------------------------
// Sphere intersection
//
// Solves the quadratic  dot(ray(t) - centre, ray(t) - centre) = radius²
// using the half-vector form for numerical stability.
//
// Returns the nearest t in (t_min, t_max), or −1 if no valid intersection.
// ---------------------------------------------------------------------------
fn sphere_hit(
    centre: vec3<f32>,
    radius: f32,
    r:      Ray,
    t_min:  f32,
    t_max:  f32,
) -> f32 {
    let oc     = r.origin - centre;
    let a      = dot(r.dir, r.dir);
    let half_b = dot(oc, r.dir);
    let c      = dot(oc, oc) - radius * radius;
    let disc   = half_b * half_b - a * c;

    if disc < 0.0 { return -1.0; }

    let sqd = sqrt(disc);

    // Try the nearer root first; fall back to the farther root.
    var t = (-half_b - sqd) / a;
    if t < t_min || t > t_max {
        t = (-half_b + sqd) / a;
        if t < t_min || t > t_max { return -1.0; }
    }
    return t;
}

// ---------------------------------------------------------------------------
// Triangle intersection  (Möller–Trumbore, double-sided)
//
// Returns the ray parameter t in (t_min, t_max), or −1 if no valid hit.
// Double-sided: both front and back faces can be hit regardless of winding.
// ---------------------------------------------------------------------------
fn triangle_hit(
    v0:    vec3<f32>,
    v1:    vec3<f32>,
    v2:    vec3<f32>,
    r:     Ray,
    t_min: f32,
    t_max: f32,
) -> f32 {
    let e1  = v1 - v0;
    let e2  = v2 - v0;
    let h   = cross(r.dir, e2);
    let det = dot(e1, h);

    // Parallel or near-degenerate triangle → no hit.
    if abs(det) < 1e-6 { return -1.0; }

    let inv_det = 1.0 / det;
    let s       = r.origin - v0;
    let u       = inv_det * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }

    let q = cross(s, e1);
    let v = inv_det * dot(r.dir, q);
    if v < 0.0 || u + v > 1.0 { return -1.0; }

    let t = inv_det * dot(e2, q);
    if t < t_min || t > t_max { return -1.0; }
    return t;
}

// ---------------------------------------------------------------------------
// World traversal  (BVH-accelerated)
//
// Delegates to hit_spheres_bvh / hit_triangles_bvh from bvh.wgsl.
// Triangle traversal carries the sphere result through so it can tighten
// the closest-t bound without a second pass.
// ---------------------------------------------------------------------------
fn hit_world(
    r:     Ray,
    t_min: f32,
    t_max: f32,
    hit:   ptr<function, Hit>,
) -> bool {
    let hit_s = hit_spheres_bvh(r, t_min, t_max, hit);
    let hit_t = hit_triangles_bvh(r, t_min, t_max, hit);
    return hit_s || hit_t;
}
