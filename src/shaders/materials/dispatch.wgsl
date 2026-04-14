// ===========================================================================
// materials/dispatch.wgsl  –  Material type constants + scatter dispatcher
//
// Depends on: common.wgsl (SphereGpu, Ray, Hit, ScatterResult)
//             materials/lambertian.wgsl, materials/metal.wgsl,
//             materials/dielectric.wgsl  (scatter_* functions)
//
// ──────────────────────────────────────────────────────────────────────────
// HOW TO ADD A NEW MATERIAL
// ──────────────────────────────────────────────────────────────────────────
//  1. Create  src/shaders/materials/<name>.wgsl
//     Define:  fn scatter_<name>(...) -> ScatterResult
//
//  2. Add a MAT_<NAME> constant below (increment the tag).
//
//  3. Add a new `case <tag>u: { ... }` to the switch inside scatter().
//
//  4. In src/raytracer.rs, add the matching include_str!() entry to the
//     concat!() block — before this file but after rng.wgsl / common.wgsl.
//
//  trace.wgsl and main.wgsl do NOT need to change.
// ===========================================================================


// ---------------------------------------------------------------------------
// Material type tags
// These must match the mat_type field values written by SphereGpu constructors
// on the Rust side (src/raytracer.rs).
// ---------------------------------------------------------------------------
const MAT_LAMBERTIAN: u32 = 0u;  // diffuse, cosine-weighted hemisphere scatter
const MAT_METAL:      u32 = 1u;  // specular reflection + optional fuzz
const MAT_DIELECTRIC: u32 = 2u;  // glass: refraction + Schlick reflection
const MAT_FRACTAL:    u32 = 3u;  // Julia-set procedural UV texture (Lambertian scatter)
// ↑ Add new MAT_* constants here as you extend the renderer.


// ---------------------------------------------------------------------------
// scatter()
//
// Central dispatcher: given the sphere that was hit and the incoming ray,
// delegates to the appropriate material scatter function and returns a
// ScatterResult.
//
// The trace loop in trace.wgsl calls this function once per bounce; it never
// needs to know which material was hit — all decisions live here.
// ---------------------------------------------------------------------------
fn scatter(
    sphere: SphereGpu,
    ray_in: Ray,
    hit:    Hit,
    rng:    ptr<function, u32>,
) -> ScatterResult {
    switch sphere.mat_type {
        case 1u: {
            // Metal — specular reflection with optional fuzz roughness.
            return scatter_metal(sphere.albedo, sphere.fuzz, ray_in, hit, rng);
        }
        case 2u: {
            // Dielectric — glass with refraction and Schlick partial reflection.
            return scatter_dielectric(sphere.ior, ray_in, hit, rng);
        }
        case 3u: {
            // Fractal — Julia-set UV texture mapped onto the sphere surface.
            // ray_in is not needed: scatter direction is purely Lambertian.
            return scatter_fractal(sphere, hit, rng);
        }
        default: {
            // Lambertian (mat_type == 0 or any unrecognised tag).
            // Cosine-weighted diffuse scatter using the surface normal.
            return scatter_lambertian(sphere.albedo, hit, rng);
        }
        // ↑ Add new `case <tag>u: { return scatter_<name>(...); }` here.
    }
}
