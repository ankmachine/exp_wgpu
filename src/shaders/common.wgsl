// =============================================================================
// common.wgsl  –  Shared types, uniforms, and GPU resource bindings
//
// Every other shader file in this module depends on the declarations here.
// Include this file first in the concat!() chain inside raytracer.rs.
// =============================================================================


// -----------------------------------------------------------------------------
// Uniforms  –  @group(0) @binding(0)
//
// Must mirror RaytracerUniforms in src/raytracer.rs exactly (96 bytes, std140).
//
// Layout – six 16-byte rows:
//   row 0  camera_pos   vec3<f32>  +  _pad0       f32
//   row 1  camera_fwd   vec3<f32>  +  _pad1       f32
//   row 2  camera_right vec3<f32>  +  fov_scale   f32   (tan(vfov/2))
//   row 3  camera_up    vec3<f32>  +  _pad2       f32
//   row 4  image_size   vec2<f32>  |  frame_count u32  |  max_bounces u32
//   row 5  lens_radius  f32        |  focus_dist  f32  |  _pad3 f32  |  _pad4 f32
// -----------------------------------------------------------------------------
struct Uniforms {
    camera_pos:   vec3<f32>,
    _pad0:        f32,
    camera_fwd:   vec3<f32>,
    _pad1:        f32,
    camera_right: vec3<f32>,
    fov_scale:    f32,        // tan(vfov / 2)
    camera_up:    vec3<f32>,
    _pad2:        f32,
    image_size:   vec2<f32>,  // (width, height) in pixels
    frame_count:  u32,        // samples accumulated so far  (0 = first frame)
    max_bounces:  u32,        // maximum path bounces before returning black
    lens_radius:  f32,        // aperture radius  (0.0 = pinhole, no DOF)
    focus_dist:   f32,        // focal-plane distance from camera origin
    _pad3:        f32,
    _pad4:        f32,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;


// -----------------------------------------------------------------------------
// Resource bindings  –  @group(1)
//
//   binding 0  accumulation buffer  –  running-mean per-pixel colour (r/w storage)
//   binding 1  display texture      –  write-only storage texture, read by display pass
//   binding 2  scene sphere array   –  read-only storage, uploaded once from the CPU
// -----------------------------------------------------------------------------
@group(1) @binding(0)
var<storage, read_write> accum: array<vec4<f32>>;

@group(1) @binding(1)
var display_tex: texture_storage_2d<rgba32float, write>;


// -----------------------------------------------------------------------------
// SphereGpu  –  48 bytes, three 16-byte rows (std430 / matches Rust SphereGpu)
//
//   offset  0  centre   vec3<f32>  (AlignOf=16 ✓)   size 12
//   offset 12  radius   f32                          size  4   → row 0 done
//   offset 16  albedo   vec3<f32>  (AlignOf=16 ✓)   size 12
//   offset 28  fuzz     f32                          size  4   → row 1 done
//   offset 32  mat_type u32                          size  4
//   offset 36  ior      f32                          size  4
//   offset 40  _pad0    f32                          size  4
//   offset 44  _pad1    f32                          size  4   → row 2 done (48 bytes total)
//
// mat_type:  0 = Lambertian  |  1 = Metal  |  2 = Dielectric
// -----------------------------------------------------------------------------
struct SphereGpu {
    centre:   vec3<f32>,
    radius:   f32,
    albedo:   vec3<f32>,  // surface colour used by Lambertian and Metal
    fuzz:     f32,        // Metal roughness: 0.0 = perfect mirror, 1.0 = fully diffuse
    mat_type: u32,        // material tag (see constants in dispatch.wgsl)
    ior:      f32,        // index of refraction (Dielectric only, e.g. 1.5 for glass)
    _pad0:    f32,
    _pad1:    f32,
}

@group(1) @binding(2)
var<storage, read> spheres: array<SphereGpu>;


// -----------------------------------------------------------------------------
// Ray
// -----------------------------------------------------------------------------
struct Ray {
    origin: vec3<f32>,
    dir:    vec3<f32>,
}

fn point_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.dir;
}


// -----------------------------------------------------------------------------
// Hit record
//
// normal always points *against* the incoming ray (front-face convention).
// sphere_idx is the index into spheres[] used to look up material data.
// -----------------------------------------------------------------------------
struct Hit {
    t:          f32,
    pos:        vec3<f32>,
    normal:     vec3<f32>,
    front:      bool,
    sphere_idx: u32,
}


// -----------------------------------------------------------------------------
// ScatterResult
//
// Returned by every material scatter function so the trace loop can stay
// generic.  did_scatter == false means the ray was absorbed (return black).
// -----------------------------------------------------------------------------
struct ScatterResult {
    scattered:   Ray,
    attenuation: vec3<f32>,
    did_scatter: bool,
}
