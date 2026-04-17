// ===========================================================================
// materials/water.wgsl  –  Water (dielectric + ripple normal perturbation)
//
// Ripples are modelled as an overlapping sine-wave height field h(x,z).
// The partial derivatives ∂h/∂x and ∂h/∂z tilt the surface normal, which
// then drives standard Schlick reflection / Snell refraction at IOR 1.33.
//
// Fields used from SphereGpu:
//   albedo  – blue-green tint applied to transmitted rays
//   ior     – refractive index (set to 1.33 by water() constructor)
//   fuzz    – ripple amplitude (0 = flat, 0.12 = noticeable waves)
//
// Depends on: common.wgsl  (Ray, Hit, ScatterResult, SphereGpu)
//             rng.wgsl     (rand)
//             dielectric.wgsl (schlick)
// ===========================================================================


// ---------------------------------------------------------------------------
// Perturb a surface normal to simulate ripples.
//
// Two sine wave trains at different angles interfere to produce a natural
// chop pattern.  The gradient of the combined height field tilts the normal.
// ---------------------------------------------------------------------------
fn water_ripple_normal(pos: vec3<f32>, normal: vec3<f32>, amplitude: f32) -> vec3<f32> {
    let freq = 2.0;

    // Two wave trains: one along X, one along a 60° diagonal.
    let w1 = pos.x * freq       + pos.z * freq * 0.5;
    let w2 = pos.x * freq * 0.5 + pos.z * freq;

    // Partial derivatives of h = sin(w1) + sin(w2), scaled by amplitude.
    let dh_dx = amplitude * (cos(w1) * freq       + cos(w2) * freq * 0.5);
    let dh_dz = amplitude * (cos(w1) * freq * 0.5 + cos(w2) * freq);

    // Subtract gradient from the normal (height bumps tilt the normal away).
    return normalize(normal + vec3<f32>(-dh_dx, 0.0, -dh_dz));
}


fn scatter_water(
    sphere: SphereGpu,
    ray_in: Ray,
    hit:    Hit,
    rng:    ptr<function, u32>,
) -> ScatterResult {
    let ior       = sphere.ior;   // 1.33 for water
    let amplitude = sphere.fuzz;  // ripple strength (0 = flat mirror)

    // Perturb normal with ripple pattern.
    let rippled_normal = water_ripple_normal(hit.pos, hit.normal, amplitude);

    // Transmitted rays pick up the water tint; reflected rays stay white.
    let tint = select(vec3<f32>(1.0), sphere.albedo, !hit.front);

    let eta       = select(ior, 1.0 / ior, hit.front);
    let unit_dir  = normalize(ray_in.dir);
    let cos_theta = min(dot(-unit_dir, rippled_normal), 1.0);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

    var dir: vec3<f32>;
    if eta * sin_theta > 1.0 || schlick(cos_theta, eta) > rand(rng) {
        dir = reflect(unit_dir, rippled_normal);
    } else {
        dir = refract(unit_dir, rippled_normal, eta);
    }

    return ScatterResult(Ray(hit.pos, dir), tint, true);
}
