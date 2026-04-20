// ===========================================================================
// materials/textured.wgsl  –  Lambertian scatter with a spherical UV texture
//
// UV mapping (equirectangular / latitude-longitude):
//   u = 0.5 + atan2(n.z, n.x) / (2π)   — longitude
//   v = 0.5 - asin(n.y)       / π       — latitude
//
// scene_texture and scene_sampler are declared in common.wgsl (bindings 4/5).
//
// Depends on: common.wgsl (SphereGpu, Hit, ScatterResult, scene_texture/sampler)
//             rng.wgsl    (rand_unit_sphere)
//             materials/lambertian.wgsl (scatter_lambertian — reused for scatter dir)
// ===========================================================================


fn sphere_uv(normal: vec3<f32>) -> vec2<f32> {
    let u = 0.5 - atan2(normal.z, normal.x) / (2.0 * 3.14159265);
    let v = 0.5 - asin(clamp(normal.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}

fn scatter_textured(
    hit: Hit,
    rng: ptr<function, u32>,
) -> ScatterResult {
    let uv      = sphere_uv(hit.normal);
    let albedo  = textureSampleLevel(scene_texture, scene_sampler, uv, 0.0).rgb;
    return scatter_lambertian(albedo, hit, rng);
}
