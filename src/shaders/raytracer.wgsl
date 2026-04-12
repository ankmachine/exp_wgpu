// ===========================================================================
// raytracer.wgsl  –  Phase 3 + 4: Materials & Depth of Field
// ===========================================================================
// Implements "Ray Tracing in One Weekend" Chapters 4 – 13:
//
//   Ch 4–6  – Rays, camera model, sphere intersection, surface normals
//   Ch 7    – Multiple objects
//   Ch 8    – Antialiasing via per-frame jittered sampling
//   Ch 9    – Lambertian (diffuse) scattering          ← from Phase 2
//   Ch 10   – Metal (specular reflection + fuzz)       ← NEW
//   Ch 11   – Dielectric (refraction + Schlick)        ← NEW
//   Ch 13   – Defocus blur (depth of field)            ← NEW
//
// Architecture
// ─────────────
//  • One compute invocation per pixel per frame.
//  • A single jittered, DOF-offset sample is traced each frame.
//  • Results are blended into a storage buffer (accum) as a running mean.
//  • The averaged linear-light value is written to a write-only storage
//    texture (display_tex) for the display pass to gamma-correct and blit.
// ===========================================================================


// ---------------------------------------------------------------------------
// Uniforms  –  @group(0) @binding(0)
// Must mirror RaytracerUniforms in src/raytracer.rs  (96 bytes, std140).
//
// Layout (six 16-byte rows):
//   row 0  camera_pos   vec3  + _pad0  f32
//   row 1  camera_fwd   vec3  + _pad1  f32
//   row 2  camera_right vec3  + fov_scale f32
//   row 3  camera_up    vec3  + _pad2  f32
//   row 4  image_size vec2 | frame_count u32 | max_bounces u32
//   row 5  lens_radius f32 | focus_dist f32 | _pad3 f32 | _pad4 f32
// ---------------------------------------------------------------------------
struct Uniforms {
    camera_pos:   vec3<f32>,
    _pad0:        f32,
    camera_fwd:   vec3<f32>,
    _pad1:        f32,
    camera_right: vec3<f32>,
    fov_scale:    f32,          // tan(vfov / 2)
    camera_up:    vec3<f32>,
    _pad2:        f32,
    image_size:   vec2<f32>,    // (width, height) in pixels
    frame_count:  u32,          // samples accumulated so far (0 = first frame)
    max_bounces:  u32,          // maximum path bounces before returning black
    lens_radius:  f32,          // aperture radius  (0 = pinhole / no DOF)
    focus_dist:   f32,          // focal-plane distance from camera origin
    _pad3:        f32,
    _pad4:        f32,
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

// Advance state and return a float uniformly in [0, 1).
// 0x2f800000 = 2^-32 as an IEEE-754 single  →  maps u32 full range to [0, 1).
fn rand(rng: ptr<function, u32>) -> f32 {
    *rng = pcg(*rng);
    return f32(*rng) * bitcast<f32>(0x2f800000u);
}

// Unit vector uniformly distributed on the sphere.
// Uses spherical coordinates – always terminates (no rejection sampling).
fn rand_unit_sphere(rng: ptr<function, u32>) -> vec3<f32> {
    let theta   = rand(rng) * 6.283185307179586;   // 2π
    let cos_phi = rand(rng) * 2.0 - 1.0;           // uniform in [−1, 1]
    let sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));
    return vec3<f32>(sin_phi * cos(theta), cos_phi, sin_phi * sin(theta));
}

// Random vector uniformly distributed *inside* the unit sphere.
// Used for Metal fuzz perturbation: the length must be ≤ 1 so the
// fuzz parameter is well-defined.  cbrt(u) gives uniform volume distribution.
fn rand_in_unit_sphere(rng: ptr<function, u32>) -> vec3<f32> {
    return rand_unit_sphere(rng) * pow(rand(rng), 0.33333333);
}

// Random point uniformly distributed inside the unit disk (xy plane).
// Used to sample the camera lens aperture for depth-of-field.
// sqrt(r) gives uniform area distribution on the disk.
fn rand_in_unit_disk(rng: ptr<function, u32>) -> vec2<f32> {
    let theta = rand(rng) * 6.283185307179586;
    let r     = sqrt(rand(rng));
    return vec2<f32>(r * cos(theta), r * sin(theta));
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
// Material types
// ===========================================================================

// 0 = Lambertian (diffuse)
// 1 = Metal      (specular reflection + optional fuzz)
// 2 = Dielectric (glass / refraction)


// ===========================================================================
// Geometry: Sphere + Hit record
// ===========================================================================

struct Sphere {
    centre:   vec3<f32>,
    radius:   f32,
    mat_type: u32,          // 0 Lambertian | 1 Metal | 2 Dielectric
    albedo:   vec3<f32>,    // surface colour (Lambertian & Metal)
    fuzz:     f32,          // Metal roughness 0 = mirror … 1 = fully diffuse
    ior:      f32,          // index of refraction (Dielectric only, e.g. 1.5)
}

struct Hit {
    t:          f32,
    pos:        vec3<f32>,
    normal:     vec3<f32>,   // always points against the incoming ray
    front:      bool,        // true → ray hit the outer (front) face
    sphere_idx: i32,         // which sphere was hit (for material lookup)
}

// Intersect a sphere; return nearest t in (t_min, t_max), or −1 on miss.
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
// Scene  (RTIOW Ch. 11 – four spheres with distinct materials)
//
//   [0]  ground  ( 0, −100.5, −1)  r=100   Lambertian  (0.8, 0.8, 0.0)
//   [1]  centre  ( 0,    0,   −1)  r=0.5   Lambertian  (0.1, 0.2, 0.5)
//   [2]  left    (−1,    0,   −1)  r=0.5   Dielectric  ior=1.5
//   [3]  right   ( 1,    0,   −1)  r=0.5   Metal       (0.8, 0.6, 0.2)  fuzz=0
//
// Phase 8 will promote this to a storage-buffer uploaded from the CPU.
// ===========================================================================

const N_SPHERES: i32 = 4;

fn scene_sphere(i: i32) -> Sphere {
    switch i {
        // Ground – large Lambertian sphere, warm yellow
        case 0: {
            return Sphere(
                vec3<f32>(0.0, -100.5, -1.0), 100.0,
                0u, vec3<f32>(0.8, 0.8, 0.0), 0.0, 0.0
            );
        }
        // Centre – Lambertian, deep blue
        case 1: {
            return Sphere(
                vec3<f32>(0.0, 0.0, -1.0), 0.5,
                0u, vec3<f32>(0.1, 0.2, 0.5), 0.0, 0.0
            );
        }
        // Left – Dielectric glass ball (ior = 1.5)
        case 2: {
            return Sphere(
                vec3<f32>(-1.0, 0.0, -1.0), 0.5,
                2u, vec3<f32>(1.0, 1.0, 1.0), 0.0, 1.5
            );
        }
        // Right – polished gold Metal
        default: {
            return Sphere(
                vec3<f32>(1.0, 0.0, -1.0), 0.5,
                1u, vec3<f32>(0.8, 0.6, 0.2), 0.0, 0.0
            );
        }
    }
}

// Walk every sphere; return the closest hit in [t_min, t_max].
// Returns false when nothing was hit (leaving *hit untouched).
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

            (*hit).t          = t;
            (*hit).pos        = p;
            (*hit).normal     = outward;
            (*hit).front      = is_front;
            (*hit).sphere_idx = i;
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
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), t);
}


// ===========================================================================
// Material scatter functions  (RTIOW Ch. 9–11)
// ===========================================================================

// Package a scatter result to avoid multiple out-pointers.
struct ScatterResult {
    scattered:   Ray,
    attenuation: vec3<f32>,
    did_scatter: bool,
}

// ── Schlick reflectance approximation (used by Dielectric) ─────────────────
fn schlick(cos_theta: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

// ── Lambertian (diffuse) ────────────────────────────────────────────────────
// Scatter direction = surface normal + random unit vector on the sphere.
// This is the exact Lambertian cosine-weighted distribution (Ch. 9).
fn scatter_lambertian(
    albedo: vec3<f32>,
    hit:    Hit,
    rng:    ptr<function, u32>,
) -> ScatterResult {
    var scatter_dir = hit.normal + rand_unit_sphere(rng);

    // Guard against a degenerate direction (normal ≈ −random).
    if abs(scatter_dir.x) < 1e-8 && abs(scatter_dir.y) < 1e-8 && abs(scatter_dir.z) < 1e-8 {
        scatter_dir = hit.normal;
    }

    return ScatterResult(Ray(hit.pos, scatter_dir), albedo, true);
}

// ── Metal (specular reflection) ─────────────────────────────────────────────
// Reflect the ray direction over the surface normal, then perturb by `fuzz`
// to simulate surface roughness.  Rays absorbed into the surface are dropped.
fn scatter_metal(
    albedo:  vec3<f32>,
    fuzz:    f32,
    ray_in:  Ray,
    hit:     Hit,
    rng:     ptr<function, u32>,
) -> ScatterResult {
    // WGSL built-in: reflect(incident, normal) → reflected direction
    let reflected   = reflect(normalize(ray_in.dir), hit.normal);
    let scatter_dir = reflected + fuzz * rand_in_unit_sphere(rng);

    // Only scatter if the perturbed ray still exits the surface.
    let did = dot(scatter_dir, hit.normal) > 0.0;
    return ScatterResult(Ray(hit.pos, scatter_dir), albedo, did);
}

// ── Dielectric (glass / refraction + Schlick) ───────────────────────────────
// Uses the WGSL built-in refract(), Schlick for partial reflection, and
// total-internal-reflection detection via Snell's law.
fn scatter_dielectric(
    ior:    f32,
    ray_in: Ray,
    hit:    Hit,
    rng:    ptr<function, u32>,
) -> ScatterResult {
    // Glass absorbs nothing – attenuation is white.
    let attenuation = vec3<f32>(1.0);

    // select(false_val, true_val, condition)
    //   front face (air → glass): eta = 1.0 / ior
    //   back face  (glass → air): eta = ior
    let eta = select(ior, 1.0 / ior, hit.front);

    let unit_dir  = normalize(ray_in.dir);
    let cos_theta = min(dot(-unit_dir, hit.normal), 1.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    var dir: vec3<f32>;

    // Total internal reflection OR Schlick probabilistic reflection.
    if eta * sin_theta > 1.0 || schlick(cos_theta, eta) > rand(rng) {
        dir = reflect(unit_dir, hit.normal);
    } else {
        // WGSL built-in: refract(incident, normal, eta)
        dir = refract(unit_dir, hit.normal, eta);
    }

    return ScatterResult(Ray(hit.pos, dir), attenuation, true);
}


// ===========================================================================
// Path tracer – iterative depth-first loop (WGSL has no recursion)
// ===========================================================================

fn trace(initial_ray: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray        = initial_ray;
    var throughput = vec3<f32>(1.0);

    for (var depth = 0i; depth < i32(u.max_bounces); depth++) {
        var hit: Hit;

        if !hit_world(ray, 0.001, 1.0e30, &hit) {
            // No geometry – sky light attenuated by accumulated throughput.
            return throughput * sky_color(ray);
        }

        let sphere = scene_sphere(hit.sphere_idx);
        var result: ScatterResult;

        // Dispatch to the correct scatter function based on material type.
        switch sphere.mat_type {
            case 1u: {   // Metal
                result = scatter_metal(sphere.albedo, sphere.fuzz, ray, hit, rng);
            }
            case 2u: {   // Dielectric
                result = scatter_dielectric(sphere.ior, ray, hit, rng);
            }
            default: {   // Lambertian  (mat_type == 0)
                result = scatter_lambertian(sphere.albedo, hit, rng);
            }
        }

        if !result.did_scatter {
            // Ray absorbed (e.g. Metal ray went below surface).
            return vec3<f32>(0.0);
        }

        throughput *= result.attenuation;
        ray         = result.scattered;
    }

    // Exceeded max depth – treat as absorbed.
    return vec3<f32>(0.0);
}


// ===========================================================================
// Compute entry-point
// ===========================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img = vec2<u32>(u32(u.image_size.x), u32(u.image_size.y));

    // Discard threads outside the image boundary (workgroup overshoot).
    if gid.x >= img.x || gid.y >= img.y { return; }

    // ── Seed RNG ─────────────────────────────────────────────────────────────
    let pixel_idx = gid.x + gid.y * img.x;
    var rng: u32  = pcg(pixel_idx ^ (u.frame_count * 2654435761u));

    // ── Jittered UV for antialiasing (RTIOW Ch. 8) ──────────────────────────
    let sx = (f32(gid.x) + rand(&rng)) / f32(img.x);
    let sy = 1.0 - (f32(gid.y) + rand(&rng)) / f32(img.y); // flip Y

    // ── Image-plane offset (fov_scale = tan(vfov/2)) ────────────────────────
    let aspect  = f32(img.x) / f32(img.y);
    let plane_x = (sx * 2.0 - 1.0) * aspect * u.fov_scale;
    let plane_y = (sy * 2.0 - 1.0) * u.fov_scale;

    // ── Depth-of-field ray generation (RTIOW Ch. 13) ────────────────────────
    //
    // The focal plane sits at distance `focus_dist` from the camera origin.
    // For each sample we offset the ray origin by a random point on the lens
    // disk and aim at the same focal-plane point, producing defocus blur.
    //
    // When lens_radius == 0 the disk offset is zero, focal_pt - camera_pos
    // reduces to focus_dist * (camera_fwd + plane_x*right + plane_y*up), and
    // normalize() cancels focus_dist → identical to the pinhole formula.
    let focal_pt = u.camera_pos
        + u.focus_dist * u.camera_fwd
        + (plane_x * u.focus_dist) * u.camera_right
        + (plane_y * u.focus_dist) * u.camera_up;

    let disk     = rand_in_unit_disk(&rng) * u.lens_radius;
    let lens_off = disk.x * u.camera_right + disk.y * u.camera_up;
    let ray_orig = u.camera_pos + lens_off;

    let ray = Ray(ray_orig, normalize(focal_pt - ray_orig));

    // ── Trace one sample ─────────────────────────────────────────────────────
    let sample = trace(ray, &rng);

    // ── Progressive accumulation (running mean) ──────────────────────────────
    // alpha == 1.0 when frame_count == 0 → full overwrite, no stale data.
    let alpha = 1.0 / f32(u.frame_count + 1u);
    let prev  = accum[pixel_idx].rgb;
    let avg   = mix(prev, sample, alpha);

    accum[pixel_idx] = vec4<f32>(avg, 1.0);
    textureStore(display_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(avg, 1.0));
}
