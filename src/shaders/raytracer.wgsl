// ===========================================================================
// raytracer.wgsl  –  Phase 2: GPU Path Tracer (Compute Shader)
// ===========================================================================
// Implements "Ray Tracing in One Weekend" Chapters 4 – 9:
//
//   Ch 4  – Rays and camera model
//   Ch 5  – Sphere intersection
//   Ch 6  – Surface normals
//   Ch 7  – Multiple objects  (four hard-coded spheres)
//   Ch 8  – Antialiasing via per-frame jittered sampling
//   Ch 9  – Lambertian (diffuse) scattering
//
// Architecture
// ─────────────
//   • One compute invocation per pixel per frame.
//   • A single jittered sample is traced and blended into a storage buffer
//     (accum) as a running mean — no explicit clear needed on reset because
//     when frame_count == 0 the blend weight is 1.0 (full overwrite).
//   • The averaged linear-light value is also written to a write-only storage
//     texture (display_tex).  The display pass (display.wgsl) reads that
//     texture, applies sqrt() gamma correction, and blits to the swapchain.
// ===========================================================================


// ---------------------------------------------------------------------------
// Uniforms  –  @group(0) @binding(0)
// Must match RaytracerUniforms in src/raytracer.rs  (80 bytes, std140).
// ---------------------------------------------------------------------------
struct Uniforms {
    camera_pos:   vec3<f32>,
    _pad0:        f32,           // keeps vec3 rows 16-byte aligned
    camera_fwd:   vec3<f32>,
    _pad1:        f32,
    camera_right: vec3<f32>,
    fov_scale:    f32,           // tan(vfov / 2)
    camera_up:    vec3<f32>,
    _pad2:        f32,
    image_size:   vec2<f32>,     // (width, height) in pixels
    frame_count:  u32,           // samples accumulated so far (0 = first frame)
    max_bounces:  u32,           // maximum path bounces before returning black
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

// ---------------------------------------------------------------------------
// Resources  –  @group(1)
//   binding 0  running-mean accumulation buffer  (storage, read_write)
//   binding 1  display texture written each frame (storage texture, write)
// ---------------------------------------------------------------------------
@group(1) @binding(0)
var<storage, read_write> accum: array<vec4<f32>>;

@group(1) @binding(1)
var display_tex: texture_storage_2d<rgba32float, write>;


// ===========================================================================
// PCG random-number generator
// Reference: https://www.pcg-random.org/
// ===========================================================================

fn pcg(v: u32) -> u32 {
    var s: u32 = v * 747796405u + 2891336453u;
    var w: u32 = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// Advance the RNG state and return a float uniformly in [0, 1).
// 0x2f800000 == 2^-32 as an IEEE-754 single, mapping u32 → [0, 1).
fn rand(rng: ptr<function, u32>) -> f32 {
    *rng = pcg(*rng);
    return f32(*rng) * bitcast<f32>(0x2f800000u);
}

// Return a unit vector uniformly distributed on the unit sphere.
// Uses spherical coordinates so it always terminates (no rejection loop).
fn rand_unit_sphere(rng: ptr<function, u32>) -> vec3<f32> {
    let theta   = rand(rng) * 6.283185307179586;   // 2π
    let cos_phi = rand(rng) * 2.0 - 1.0;           // uniform in [−1, 1]
    let sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));
    return vec3<f32>(sin_phi * cos(theta), cos_phi, sin_phi * sin(theta));
}


// ===========================================================================
// Ray
// ===========================================================================

struct Ray {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

fn point_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.dir;
}


// ===========================================================================
// Geometry: Sphere + Hit record
// ===========================================================================

struct Sphere {
    centre: vec3<f32>,
    radius: f32,
}

struct Hit {
    t:      f32,
    pos:    vec3<f32>,
    normal: vec3<f32>,   // always points against the incoming ray
    front:  bool,        // true  → ray hit the outer face
}

// Intersect a sphere and return the nearest t in (t_min, t_max), or -1 on miss.
fn sphere_hit(s: Sphere, r: Ray, t_min: f32, t_max: f32) -> f32 {
    let oc     = r.origin - s.centre;
    let a      = dot(r.dir, r.dir);
    let half_b = dot(oc, r.dir);
    let c      = dot(oc, oc) - s.radius * s.radius;
    let disc   = half_b * half_b - a * c;

    if disc < 0.0 { return -1.0; }

    let sqd = sqrt(disc);
    var t   = (-half_b - sqd) / a;
    if t < t_min || t > t_max {
        t = (-half_b + sqd) / a;
        if t < t_min || t > t_max { return -1.0; }
    }
    return t;
}


// ===========================================================================
// Scene  (RTIOW Ch. 7 – four hard-coded spheres)
//
//   [0]  centre ball  ( 0,     0,  −1)   r = 0.5
//   [1]  ground       ( 0, −100.5, −1)   r = 100
//   [2]  left ball    (−1,     0,  −1)   r = 0.5
//   [3]  right ball   ( 1,     0,  −1)   r = 0.5
//
// Phase 8 will replace this with a storage-buffer scene uploaded from the CPU.
// ===========================================================================

const N_SPHERES: i32 = 4;

fn scene_sphere(i: i32) -> Sphere {
    switch i {
        case 0:  { return Sphere(vec3<f32>( 0.0,    0.0, -1.0),   0.5); }
        case 1:  { return Sphere(vec3<f32>( 0.0, -100.5, -1.0), 100.0); }
        case 2:  { return Sphere(vec3<f32>(-1.0,    0.0, -1.0),   0.5); }
        default: { return Sphere(vec3<f32>( 1.0,    0.0, -1.0),   0.5); }
    }
}

// Walk every sphere and return the closest hit in [t_min, t_max].
// Returns false (and leaves *hit untouched) when nothing was hit.
fn hit_world(r: Ray, t_min: f32, t_max: f32, hit: ptr<function, Hit>) -> bool {
    var closest = t_max;
    var found   = false;

    for (var i = 0i; i < N_SPHERES; i++) {
        let s = scene_sphere(i);
        let t = sphere_hit(s, r, t_min, closest);

        if t > 0.0 {
            closest = t;
            found   = true;

            let p        = point_at(r, t);
            var outward  = (p - s.centre) / s.radius;
            let is_front = dot(r.dir, outward) < 0.0;
            if !is_front { outward = -outward; }

            (*hit).t      = t;
            (*hit).pos    = p;
            (*hit).normal = outward;
            (*hit).front  = is_front;
        }
    }
    return found;
}


// ===========================================================================
// Background / sky colour  (RTIOW Ch. 4)
// ===========================================================================

fn sky_color(r: Ray) -> vec3<f32> {
    let unit = normalize(r.dir);
    let t    = 0.5 * (unit.y + 1.0);
    // Blend: white (t = 0, top) → sky-blue (t = 1, bottom)
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), t);
}


// ===========================================================================
// Path tracer – iterative (WGSL has no recursion)
// Implements Lambertian diffuse scattering (RTIOW Ch. 9).
// ===========================================================================

fn trace(initial_ray: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray        = initial_ray;
    var throughput = vec3<f32>(1.0);

    for (var depth = 0i; depth < i32(u.max_bounces); depth++) {
        var hit: Hit;

        if !hit_world(ray, 0.001, 1.0e30, &hit) {
            // No geometry hit – attenuate sky colour and return.
            return throughput * sky_color(ray);
        }

        // Lambertian diffuse: scatter randomly in the hemisphere around normal.
        var scatter = hit.normal + rand_unit_sphere(rng);

        // Guard against a near-zero scatter vector (normal ≈ −random).
        if abs(scatter.x) < 1e-8 && abs(scatter.y) < 1e-8 && abs(scatter.z) < 1e-8 {
            scatter = hit.normal;
        }

        ray        = Ray(hit.pos, scatter);
        throughput *= 0.5;   // 50 % grey albedo for all surfaces in this phase
    }

    // Exceeded max depth – ray is absorbed; contribute black.
    return vec3<f32>(0.0);
}


// ===========================================================================
// Compute entry-point
// ===========================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img = vec2<u32>(u32(u.image_size.x), u32(u.image_size.y));

    // Discard threads that fall outside the image (workgroup boundary overshoot).
    if gid.x >= img.x || gid.y >= img.y { return; }

    // ── Seed the RNG ────────────────────────────────────────────────────────
    // XOR the flat pixel index with a frame-dependent scramble so every frame
    // draws a different jitter sample.  Multiplying by a large odd constant
    // decorrelates successive frame seeds.
    let pixel_idx = gid.x + gid.y * img.x;
    var rng: u32  = pcg(pixel_idx ^ (u.frame_count * 2654435761u));

    // ── Jittered camera ray (RTIOW Ch. 8 antialiasing) ──────────────────────
    // Add a sub-pixel random offset so each frame samples a slightly different
    // point within the pixel footprint.
    let sx = (f32(gid.x) + rand(&rng)) / f32(img.x);        // [0, 1)
    let sy = 1.0 - (f32(gid.y) + rand(&rng)) / f32(img.y);  // flip Y: screen↓ → world↑

    // Map UV → image-plane offset.  fov_scale = tan(vfov / 2).
    let aspect  = f32(img.x) / f32(img.y);
    let plane_x = (sx * 2.0 - 1.0) * aspect * u.fov_scale;
    let plane_y = (sy * 2.0 - 1.0) * u.fov_scale;

    let ray = Ray(
        u.camera_pos,
        normalize(u.camera_fwd + plane_x * u.camera_right + plane_y * u.camera_up),
    );

    // ── Trace ────────────────────────────────────────────────────────────────
    let sample = trace(ray, &rng);

    // ── Progressive accumulation – running mean ──────────────────────────────
    // When frame_count == 0, alpha == 1.0, so the old (possibly stale) value
    // in the buffer is completely overwritten.  This makes explicit buffer
    // clearing unnecessary on camera-move / resize resets – just set
    // frame_count back to 0 in Rust and the first new frame takes over cleanly.
    let alpha = 1.0 / f32(u.frame_count + 1u);
    let prev  = accum[pixel_idx].rgb;
    let avg   = mix(prev, sample, alpha);

    // Persist the running mean in the accumulation buffer …
    accum[pixel_idx] = vec4<f32>(avg, 1.0);

    // … and write the current average to the display texture.
    // display.wgsl will apply sqrt() gamma correction before presenting.
    textureStore(display_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(avg, 1.0));
}
