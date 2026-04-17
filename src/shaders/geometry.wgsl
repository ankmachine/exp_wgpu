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
// World traversal
//
// Walks every sphere in the `spheres` storage buffer and returns the closest
// hit in [t_min, t_max].  Returns false when nothing was hit and leaves
// *hit untouched.
//
// The outward normal is always flipped to face against the incoming ray so
// material functions never have to worry about which side they are on.
// ---------------------------------------------------------------------------
fn hit_world(
    r:    Ray,
    t_min: f32,
    t_max: f32,
    hit:  ptr<function, Hit>,
) -> bool {
    let n_spheres   = arrayLength(&spheres);
    let n_triangles = arrayLength(&triangles);
    var closest     = t_max;
    var found       = false;

    // --- Spheres ---
    for (var i = 0u; i < n_spheres; i++) {
        let s = spheres[i];
        let t = sphere_hit(s.centre, s.radius, r, t_min, closest);

        if t > 0.0 {
            closest = t;
            found   = true;

            let p        = point_at(r, t);
            var outward  = (p - s.centre) / s.radius;  // unit outward normal
            let is_front = dot(r.dir, outward) < 0.0;
            if !is_front { outward = -outward; }

            (*hit).t           = t;
            (*hit).pos         = p;
            (*hit).normal      = outward;
            (*hit).front       = is_front;
            (*hit).sphere_idx  = i;
            (*hit).is_triangle = 0u;
            (*hit).tri_idx     = 0u;
        }
    }

    // --- Triangles ---
    for (var i = 0u; i < n_triangles; i++) {
        let tri = triangles[i];
        let t   = triangle_hit(tri.v0, tri.v1, tri.v2, r, t_min, closest);

        if t > 0.0 {
            closest = t;
            found   = true;

            let p        = point_at(r, t);
            let e1       = tri.v1 - tri.v0;
            let e2       = tri.v2 - tri.v0;
            var outward  = normalize(cross(e1, e2));
            let is_front = dot(r.dir, outward) < 0.0;
            if !is_front { outward = -outward; }

            (*hit).t           = t;
            (*hit).pos         = p;
            (*hit).normal      = outward;
            (*hit).front       = is_front;
            (*hit).sphere_idx  = 0u;
            (*hit).is_triangle = 1u;
            (*hit).tri_idx     = i;
        }
    }
    return found;
}
