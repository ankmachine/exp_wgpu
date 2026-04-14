// ===========================================================================
// materials/fractal.wgsl  –  Julia-set procedural surface material
//
// Depends on:  common.wgsl  (Ray, Hit, ScatterResult, SphereGpu)
//              rng.wgsl     (rand_unit_sphere)
//
// Maps the Julia set  z_{n+1} = z_n² + c  onto the sphere surface using
// spherical UV coordinates, producing a vivid procedural fractal colour
// pattern.  Rays scatter diffusely (Lambertian) with the fractal colour
// as the surface albedo.
//
// SphereGpu field mapping (repurposed for fractal parameters):
//
//   albedo.r, albedo.g  →  Julia constant c  (real and imaginary parts)
//                          Some beautiful values to try:
//                            -0.70,  0.27   "dragon"   – intricate tendrils
//                             0.28,  0.01   "tree"     – branching structure
//                            -0.40,  0.60   "dust"     – disconnected islands
//                            -0.54,  0.54   "spiral"   – tight swirling arms
//                            -0.12, -0.77   "sun"      – radial symmetry
//
//   albedo.b            →  colour palette index (float, floored to u32 % 4)
//                            0 = Rainbow   full-spectrum cycle
//                            1 = Magma     deep reds and oranges
//                            2 = Ice       cool blues and cyans
//                            3 = Gold      warm golds and bronzes
//
//   fuzz                →  UV scale / zoom factor (try 1.5 – 3.0; default 2.0)
//                          Larger values zoom out and show more of the fractal.
//
//   ior                 →  max iteration depth, stored as f32, cast to i32
//                          in the shader (try 32 – 128; default 64)
//                          Higher values reveal finer detail but cost more GPU time.
// ===========================================================================


// ---------------------------------------------------------------------------
// Cosine-basis colour palettes  (technique by Inigo Quilez)
//
//   color(t) = a + b * cos(2π * (c * t + d))
//
// Each channel (r, g, b) is modulated independently, giving smooth periodic
// colour cycles across the fractal's escape-time gradient.
// ---------------------------------------------------------------------------
fn fractal_palette(t: f32, palette_id: u32) -> vec3<f32> {
    var a:   vec3<f32>;
    var b:   vec3<f32>;
    var frq: vec3<f32>;   // frequency  (named frq to avoid shadowing Julia c)
    var d:   vec3<f32>;   // phase offset

    switch palette_id % 4u {

        case 1u: {  // ── Magma  (deep reds, oranges, near-black interiors)
            a   = vec3<f32>(0.50, 0.20, 0.10);
            b   = vec3<f32>(0.50, 0.40, 0.30);
            frq = vec3<f32>(1.00, 0.70, 0.40);
            d   = vec3<f32>(0.00, 0.15, 0.20);
        }

        case 2u: {  // ── Ice  (cold blues, cyans, soft whites)
            a   = vec3<f32>(0.10, 0.30, 0.50);
            b   = vec3<f32>(0.40, 0.40, 0.50);
            frq = vec3<f32>(0.50, 0.80, 1.00);
            d   = vec3<f32>(0.00, 0.10, 0.20);
        }

        case 3u: {  // ── Gold  (warm golds, bronzes, amber)
            a   = vec3<f32>(0.80, 0.60, 0.10);
            b   = vec3<f32>(0.20, 0.30, 0.20);
            frq = vec3<f32>(2.00, 1.50, 1.00);
            d   = vec3<f32>(0.00, 0.25, 0.50);
        }

        default: { // ── Rainbow  (full-spectrum, palette_id == 0 or unrecognised)
            a   = vec3<f32>(0.50, 0.50, 0.50);
            b   = vec3<f32>(0.50, 0.50, 0.50);
            frq = vec3<f32>(1.00, 1.00, 1.00);
            d   = vec3<f32>(0.00, 0.33, 0.67);
        }
    }

    return clamp(a + b * cos(6.283185307 * (frq * t + d)), vec3<f32>(0.0), vec3<f32>(1.0));
}


// ---------------------------------------------------------------------------
// Julia-set iteration with smooth escape-time colouring
//
// Iterates  z_{n+1} = z_n² + c  starting from  z_0 = uv.
//
// Returns:
//   0.0           uv is inside the Julia set  (orbit never escaped)
//   (0.0, 1.0)    smooth escape time for exterior points
//
// Smooth colouring  (removes the harsh colour banding of integer counts):
//
//   mu = N - log₂(log₂(|z_N|))
//
// where N is the first escape iteration and |z_N| is the magnitude at N.
// Derivation:
//   log₂(|z|)      = 0.5 · ln(|z|²) / ln(2)
//   log₂(log₂(|z|)) = ln( log₂(|z|) ) / ln(2)
//
// At exact escape (|z| = 2) the correction is 0, so mu = N exactly.
// As |z| grows the correction subtracts a fraction, smoothing the bands.
// ---------------------------------------------------------------------------
fn julia_smooth(uv: vec2<f32>, c: vec2<f32>, max_iters: i32) -> f32 {
    var z = uv;
    var i = 0;

    while i < max_iters && dot(z, z) < 4.0 {
        z = vec2<f32>(z.x * z.x - z.y * z.y + c.x,
                      2.0 * z.x * z.y           + c.y);
        i += 1;
    }

    // Interior of the Julia set — the orbit is bounded forever.
    if i >= max_iters { return 0.0; }

    // Smooth colouring for escaped orbits.
    // Clamp mag2 away from the escape boundary to keep the logs well-defined.
    let ln2  = 0.6931471806;               // ln(2)
    let mag2 = max(dot(z, z), 4.001);      // |z|² at escape  (always ≥ 4)
    let log2_mag      = 0.5 * log(mag2) / ln2;          // log₂(|z|)
    let log2_log2_mag = log(max(log2_mag, 0.001)) / ln2; // log₂(log₂(|z|))

    let smooth_i = f32(i) - log2_log2_mag;
    return clamp(smooth_i / f32(max_iters), 0.0, 1.0);
}


// ---------------------------------------------------------------------------
// scatter_fractal
//
// Pipeline:
//   1. Convert sphere surface normal → spherical UV coordinates (u, v).
//   2. Scale UV into the complex plane using the zoom parameter.
//   3. Evaluate the Julia set; compute a smooth escape-time value t ∈ [0, 1].
//   4. Map t through the chosen cosine palette to get a linear-light colour.
//   5. Scatter the ray diffusely (Lambertian) using that colour as albedo.
//
// Note on the UV seam: atan2 wraps at ±π, which creates a 1-texel seam at
// the meridian facing away from positive-X.  For complex fractal patterns
// this discontinuity is visually imperceptible.
// ---------------------------------------------------------------------------
fn scatter_fractal(
    sphere: SphereGpu,
    hit:    Hit,
    rng:    ptr<function, u32>,
) -> ScatterResult {

    // ── Unpack fractal parameters from SphereGpu fields ─────────────────────
    let julia_c   = sphere.albedo.xy;           // Julia constant (real, imag)
    let palette   = u32(sphere.albedo.z);       // colour palette selector
    let zoom      = max(sphere.fuzz, 0.5);      // UV scale  (guard against 0)
    let max_iters = max(i32(sphere.ior), 8);    // iteration depth (min 8)

    // ── Spherical UV mapping ─────────────────────────────────────────────────
    // The hit normal is the unit vector from the sphere centre to the hit point,
    // i.e. a point on the unit sphere.  We convert it to (u, v) ∈ [0,1]²:
    //
    //   u  =  normalised azimuth    atan2(z, x) mapped to [0, 1)
    //   v  =  normalised elevation  asin(y)     mapped to [0, 1]
    //         (v=0 at north pole, v=1 at south pole)
    let n      = hit.normal;
    let pi     = 3.14159265359;
    let two_pi = 6.28318530718;

    let u_coord = 0.5 + atan2(n.z, n.x) / two_pi;
    let v_coord = 0.5 - asin(clamp(n.y, -1.0, 1.0)) / pi;

    // ── Map UV into the complex plane ────────────────────────────────────────
    // Centre on (0.5, 0.5) then scale so the interesting region of the Julia
    // set is visible across the sphere surface.
    let uv = (vec2<f32>(u_coord, v_coord) - vec2<f32>(0.5)) * zoom * 2.5;

    // ── Julia-set evaluation ─────────────────────────────────────────────────
    let t = julia_smooth(uv, julia_c, max_iters);

    // ── Colour mapping ───────────────────────────────────────────────────────
    // Points inside the set (t == 0.0) are rendered near-black so the fractal
    // shape itself is visible as a dark silhouette against the coloured exterior.
    // A tiny non-zero albedo ensures they still participate in global illumination
    // rather than acting as perfect absorbers.
    var color: vec3<f32>;
    if t <= 0.0 {
        color = vec3<f32>(0.015, 0.015, 0.025);   // deep indigo-black interior
    } else {
        color = fractal_palette(t, palette);
    }

    // ── Lambertian diffuse scatter ───────────────────────────────────────────
    var scatter_dir = hit.normal + rand_unit_sphere(rng);

    // Guard: if the random vector is nearly opposite to the normal the sum
    // can be near-zero.  Fall back to the normal to keep the direction valid.
    if abs(scatter_dir.x) < 1e-8 &&
       abs(scatter_dir.y) < 1e-8 &&
       abs(scatter_dir.z) < 1e-8 {
        scatter_dir = hit.normal;
    }

    return ScatterResult(Ray(hit.pos, scatter_dir), color, true);
}
