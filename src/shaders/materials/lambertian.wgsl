// ===========================================================================
// materials/lambertian.wgsl  –  Lambertian (diffuse) scattering  (RTIOW Ch. 9)
//
// Depends on: common.wgsl (Hit, Ray, ScatterResult), rng.wgsl (rand_unit_sphere)
// ===========================================================================

/// Scatter a ray diffusely from a Lambertian surface.
///
/// The scatter direction is the surface normal plus a random unit vector, which
/// produces a cosine-weighted distribution over the hemisphere — this is the
/// exact Lambertian BRDF, not merely an approximation.
///
/// `albedo`  surface reflectance colour  (linear light, each channel in [0, 1])
/// `hit`     the surface hit record
/// `rng`     PCG state pointer threaded from the compute entry point
fn scatter_lambertian(
    albedo: vec3<f32>,
    hit:    Hit,
    rng:    ptr<function, u32>,
) -> ScatterResult {
    var scatter_dir = hit.normal + rand_unit_sphere(rng);

    // Guard against a near-zero scatter direction that arises when the random
    // vector is almost exactly opposite to the surface normal.  Falling back
    // to the normal itself keeps the direction valid and well-normalised.
    if abs(scatter_dir.x) < 1e-8 &&
       abs(scatter_dir.y) < 1e-8 &&
       abs(scatter_dir.z) < 1e-8 {
        scatter_dir = hit.normal;
    }

    return ScatterResult(Ray(hit.pos, scatter_dir), albedo, true);
}
