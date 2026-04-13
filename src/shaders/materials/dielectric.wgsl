// ===========================================================================
// materials/dielectric.wgsl  –  Dielectric (glass / refraction)  (RTIOW Ch. 11)
//
// Depends on: common.wgsl  (Ray, Hit, ScatterResult)
//             rng.wgsl     (rand)
// ===========================================================================


// ---------------------------------------------------------------------------
// Schlick reflectance approximation
//
// Approximates the Fresnel equations: at glancing angles, even glass reflects.
// `cos_theta` is the cosine of the angle between the ray and the surface normal.
// `ref_idx`   is the ratio of refractive indices (n_i / n_t).
// ---------------------------------------------------------------------------
fn schlick(cos_theta: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}


// ---------------------------------------------------------------------------
// scatter_dielectric
//
// Models a perfectly smooth glass-like surface:
//   • Glass absorbs nothing  →  attenuation is always white (1, 1, 1).
//   • Uses Snell's law (WGSL built-in refract) for transmitted rays.
//   • Uses Schlick to stochastically choose reflection vs. refraction at
//     glancing angles, and handles total internal reflection.
//
// The refraction ratio flips automatically for front- vs. back-face hits:
//   front face (air → glass):  eta = 1.0 / ior
//   back face  (glass → air):  eta = ior
//
// select(false_value, true_value, condition) is the WGSL ternary.
// ---------------------------------------------------------------------------
fn scatter_dielectric(
    ior:    f32,
    ray_in: Ray,
    hit:    Hit,
    rng:    ptr<function, u32>,
) -> ScatterResult {
    let attenuation = vec3<f32>(1.0);   // glass absorbs nothing

    // Ratio of incident to transmitted refractive index.
    let eta = select(ior, 1.0 / ior, hit.front);

    let unit_dir  = normalize(ray_in.dir);
    let cos_theta = min(dot(-unit_dir, hit.normal), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    // Total internal reflection occurs when sin(theta_t) > 1 (Snell's law).
    // Schlick adds probabilistic partial reflection at all other angles.
    var dir: vec3<f32>;
    if eta * sin_theta > 1.0 || schlick(cos_theta, eta) > rand(rng) {
        // Reflect — either TIR or Schlick-weighted Fresnel reflection.
        dir = reflect(unit_dir, hit.normal);
    } else {
        // Refract — WGSL built-in: refract(incident, normal, eta)
        dir = refract(unit_dir, hit.normal, eta);
    }

    return ScatterResult(Ray(hit.pos, dir), attenuation, true);
}
