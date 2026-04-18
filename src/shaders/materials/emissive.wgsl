// ===========================================================================
// materials/emissive.wgsl  –  Diffuse area light  (RTNW Ch. 7)
//
// An emissive surface does not scatter rays — it terminates the path and
// contributes its emitted colour weighted by the accumulated throughput.
// The trace loop treats attenuation-when-!did_scatter as emitted radiance.
//
// albedo holds the emitted colour; multiply by a brightness > 1 to make the
// light bright enough to illuminate the scene (e.g. [5, 5, 5] = 5× white).
//
// Depends on: common.wgsl  (Ray, Hit, ScatterResult, SphereGpu)
// ===========================================================================


fn scatter_emissive(sphere: SphereGpu, hit: Hit) -> ScatterResult {
    // Only emit on the front face; back face is dark (one-sided light).
    let emitted = select(vec3<f32>(0.0), sphere.albedo, hit.front);
    return ScatterResult(Ray(hit.pos, hit.normal), emitted, false);
}
