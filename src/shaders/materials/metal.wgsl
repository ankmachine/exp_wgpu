// ===========================================================================
// materials/metal.wgsl  –  Metal (specular reflection + fuzz)  RTIOW Ch. 10
//
// Depends on: common.wgsl  (Ray, Hit, ScatterResult)
//             rng.wgsl     (rand_in_unit_sphere)
// ===========================================================================

/// Scatter a ray off a metallic surface.
///
/// The ray is reflected about the surface normal and then perturbed by a
/// random vector inside the unit sphere scaled by `fuzz`.
///
///   fuzz = 0.0  →  perfect mirror
///   fuzz = 1.0  →  maximally rough (diffuse-like appearance)
///
/// If the perturbed direction points *into* the surface (dot product ≤ 0)
/// the ray is considered absorbed and `did_scatter` is set to false.
fn scatter_metal(
    albedo:  vec3<f32>,
    fuzz:    f32,
    ray_in:  Ray,
    hit:     Hit,
    rng:     ptr<function, u32>,
) -> ScatterResult {
    // WGSL built-in reflect(incident, normal) computes the mirrored direction.
    let reflected   = reflect(normalize(ray_in.dir), hit.normal);
    let scatter_dir = reflected + fuzz * rand_in_unit_sphere(rng);

    // Absorb the ray if the fuzz perturbation pushed it below the surface.
    let did = dot(scatter_dir, hit.normal) > 0.0;
    return ScatterResult(Ray(hit.pos, scatter_dir), albedo, did);
}
