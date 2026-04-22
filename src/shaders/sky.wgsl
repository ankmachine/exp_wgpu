// ===========================================================================
// sky.wgsl  –  Background / sky colour
//
// Depends on: common.wgsl (Ray)
// ===========================================================================

/// Returns the sky background colour for a ray that missed all geometry.
/// Blends white (ray pointing up) → sky-blue (ray pointing down),
/// matching the RTIOW gradient exactly.
fn sky_color(r: Ray) -> vec3<f32> {
    let unit = normalize(r.dir);
    let t    = 0.1 * (unit.y + 1.0);
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), t) * 0.00005;
    // return vec3<f32>(0.0);
}
