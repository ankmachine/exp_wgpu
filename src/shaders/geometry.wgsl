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
    let n_spheres = arrayLength(&spheres);
    var closest   = t_max;
    var found     = false;

    for (var i = 0u; i < n_spheres; i++) {
        let s = spheres[i];
        let t = sphere_hit(s.centre, s.radius, r, t_min, closest);

        if t > 0.0 {
            closest = t;
            found   = true;

            let p        = point_at(r, t);
            var outward  = (p - s.centre) / s.radius;  // unit outward normal
            let is_front = dot(r.dir, outward) < 0.0;
            if !is_front { outward = -outward; }        // flip for back-face hits

            (*hit).t          = t;
            (*hit).pos        = p;
            (*hit).normal     = outward;
            (*hit).front      = is_front;
            (*hit).sphere_idx = i;
        }
    }
    return found;
}
