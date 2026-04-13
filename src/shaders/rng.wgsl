// ===========================================================================
// rng.wgsl  –  PCG random-number generator + distribution samplers
// ===========================================================================
// All functions are stateless except for the `rng: ptr<function, u32>` state
// pointer that is threaded through the call chain.  Seed uniquely per pixel
// and per frame in the compute entry point (main.wgsl).
//
// Reference: https://www.pcg-random.org/
// ===========================================================================


// ---------------------------------------------------------------------------
// Core PCG hash
// One application produces a high-quality pseudo-random u32 from any u32 input.
// ---------------------------------------------------------------------------
fn pcg(v: u32) -> u32 {
    var s: u32 = v * 747796405u + 2891336453u;
    var w: u32 = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}


// ---------------------------------------------------------------------------
// Uniform float in [0, 1)
// 0x2f800000 == 2^-32 as an IEEE-754 single-precision float.
// Multiplying a u32 by 2^-32 maps the full u32 range onto [0, 1).
// ---------------------------------------------------------------------------
fn rand(rng: ptr<function, u32>) -> f32 {
    *rng = pcg(*rng);
    return f32(*rng) * bitcast<f32>(0x2f800000u);
}


// ---------------------------------------------------------------------------
// Uniform unit vector on the surface of the unit sphere.
// Uses spherical coordinates – always terminates, no rejection sampling needed.
//   θ  uniform in [0, 2π)
//   φ  uniform in [0, π) via cosφ uniform in [-1, 1]
// ---------------------------------------------------------------------------
fn rand_unit_sphere(rng: ptr<function, u32>) -> vec3<f32> {
    let theta   = rand(rng) * 6.283185307179586;        // 2π
    let cos_phi = rand(rng) * 2.0 - 1.0;               // uniform in [−1, 1]
    let sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));
    return vec3<f32>(sin_phi * cos(theta), cos_phi, sin_phi * sin(theta));
}


// ---------------------------------------------------------------------------
// Uniform random vector *inside* the unit sphere (length ≤ 1).
// Used for Metal fuzz: the fuzz radius must be bounded so that fuzz=1
// produces a well-defined worst-case roughness.
//
// Method: take a unit direction and scale by r = u^(1/3) so the radial CDF
// matches the volume of a sphere (V ∝ r³  →  CDF ∝ r³  →  r = u^(1/3)).
// ---------------------------------------------------------------------------
fn rand_in_unit_sphere(rng: ptr<function, u32>) -> vec3<f32> {
    return rand_unit_sphere(rng) * pow(rand(rng), 0.33333333);
}


// ---------------------------------------------------------------------------
// Uniform random point inside the unit disk (the xy-plane, z = 0).
// Used to jitter the ray origin across the camera lens aperture for DOF.
//
// Method: polar coordinates with r = sqrt(u) for uniform area distribution
// (area element = r dr dθ  →  CDF ∝ r²  →  r = sqrt(u)).
// ---------------------------------------------------------------------------
fn rand_in_unit_disk(rng: ptr<function, u32>) -> vec2<f32> {
    let theta = rand(rng) * 6.283185307179586;
    let r     = sqrt(rand(rng));
    return vec2<f32>(r * cos(theta), r * sin(theta));
}
