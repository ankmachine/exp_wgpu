// ===========================================================================
// trace.wgsl  –  Iterative path tracer  (WGSL has no recursion)
//
// Depends on: common.wgsl  (Ray, Hit, ScatterResult, u, spheres)
//             geometry.wgsl (hit_world)
//             sky.wgsl      (sky_color)
//             materials/dispatch.wgsl (scatter)
//
// This file never needs to change when adding new materials.  All material
// routing is handled by scatter() in materials/dispatch.wgsl.
// ===========================================================================

/// Trace `initial_ray` through the scene and return a linear-light RGB colour.
///
/// The loop iterates until either:
///   • the ray escapes to the sky  (return sky colour × accumulated throughput)
///   • the ray is absorbed         (return black)
///   • max_bounces is reached      (return black — energy conservation cutoff)
///
/// `initial_ray`  the primary (or secondary) ray to trace
/// `rng`          PCG state pointer; advanced by every scatter / RNG call
fn trace(initial_ray: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray        = initial_ray;
    var throughput = vec3<f32>(1.0);

    for (var depth = 0i; depth < i32(u.max_bounces); depth++) {
        var hit: Hit;

        if !hit_world(ray, 0.001, 1.0e30, &hit) {
            // Ray escaped — weight sky luminance by accumulated throughput.
            return throughput * sky_color(ray);
        }

        // Look up the material for the hit primitive and scatter the ray.
        let result = scatter_from_hit(hit, ray, rng);

        if !result.did_scatter {
            // Attenuation doubles as emitted radiance when did_scatter=false.
            // Skip on depth==0 so the light sphere is invisible to primary rays
            // but still illuminates the scene via indirect bounces.
            if depth == 0i {
                return vec3<f32>(0.0);
            }
            return throughput * result.attenuation;
        }

        // Accumulate attenuation and continue with the scattered ray.
        throughput *= result.attenuation;
        ray         = result.scattered;
    }

    // Exceeded the maximum bounce depth — treat the remaining energy as lost.
    return vec3<f32>(0.0);
}
