Learning wgpu


steps
1. Create instance to create Adapter and Surface.
2. Adapter handle graphic card. We create Device and Queue from it.
3. Get Pipeline setup.
4. Have Vertex and indices now.
5. Camera moves with controller.

##
Current Status:

Render triangle, and move camera

##
Raytracing Implementation plan:

Plan: Ray Tracing in One Weekend on the GPU

The core architectural shift is: **replace the rasterizer with a GPU compute ray tracer + a display pass.** Here's how every piece maps to your current code:

---

### Architecture Overview

```/dev/null/architecture.txt#L1-20
┌─────────────────────────────────────────────────────────┐
│  Each Frame                                             │
│                                                         │
│  1. COMPUTE PASS  (raytracer.wgsl)                      │
│     ├── 1 thread per pixel                              │
│     ├── Reads  accumulation_texture (RGBA32Float)       │
│     ├── Traces one ray, scatters, shades                │
│     └── Writes blended result → accumulation_texture   │
│                                                         │
│  2. RENDER PASS   (display.wgsl)                        │
│     ├── Fullscreen triangle                             │
│     ├── Samples accumulation_texture                    │
│     ├── Gamma-corrects (sqrt)                           │
│     └── Writes → swapchain surface                     │
└─────────────────────────────────────────────────────────┘
```

---

### New File Layout

```/dev/null/layout.txt#L1-20
src/
  main.rs              ← unchanged
  lib.rs               ← modified: drive both passes, handle resize/reset
  camera.rs            ← kept, extended with ray-gen helpers
  vertex.rs            ← repurposed for fullscreen triangle
  raytracer.rs         ← NEW: builds compute pipeline, manages textures & uniforms
  shaders/
    shader.wgsl        ← replaced with display.wgsl role
    raytracer.wgsl     ← NEW: the full ray tracer compute shader
    display.wgsl       ← NEW: fullscreen blit + gamma correction
```

---

### Phase-by-Phase Implementation

#### Phase 1 — Display Infrastructure
**Goal:** Get a fullscreen quad reading from a texture and displaying it.

- Add a `RGBA32Float` storage texture (`accumulation_texture`) sized to the window
- Create a **display render pipeline** with a fullscreen triangle (1 triangle, no vertex buffer needed)
- Write `display.wgsl`: sample `accumulation_texture`, gamma correct (`sqrt(color)`), output to swapchain
- On window resize: recreate `accumulation_texture` and reset frame count

#### Phase 2 — Compute Pipeline Scaffold + Uniforms
**Goal:** Wire up the compute pass that will do the ray tracing.

- Create `src/raytracer.rs` with a `RaytracerPipeline` struct
- Define `RaytracerUniforms` (sent as a uniform buffer each frame):
```/dev/null/uniforms.rs#L1-15
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RaytracerUniforms {
    camera_pos:    [f32; 3], _pad0: f32,
    camera_fwd:    [f32; 3], _pad1: f32,
    camera_right:  [f32; 3], _pad2: f32,
    camera_up:     [f32; 3], fov_scale: f32,
    image_size:    [f32; 2],
    frame_count:   u32,
    max_bounces:   u32,
    lens_radius:   f32,      focus_dist: f32,
    _pad3:         [f32; 2],
}
```
- Bind layout: `@group(0)` = uniforms, `@group(1)` = accumulation texture (read+write)
- Dispatch: `ceil(width/8) × ceil(height/8)` workgroups

#### Phase 3 — RTIOW Chapters 4–6: Rays, Background, Sphere
**Goal:** See a ray-colored sphere on screen.

In `raytracer.wgsl`:
- `Ray { origin, direction }` struct
- `ray_at(ray, t)` helper
- Background gradient (lerp white→sky-blue on Y)
- `hit_sphere(center, radius, ray, t_min, t_max)` → `HitRecord`
- Color normals as RGB

#### Phase 4 — Chapter 7: Multiple Objects (Hit List)
- `Sphere` array hardcoded in the shader (WGSL array literals)
- `hit_world(ray, t_min, t_max)` loops all spheres, returns closest hit

#### Phase 5 — Chapter 8: Antialiasing + Accumulation
**Goal:** Progressive refinement — each frame adds one more sample.

- PCG hash PRNG seeded by `pixel_coord + frame_count * large_prime`
- Each compute invocation: jitter the ray within the pixel
- Blend result: `accum = (accum * frame + new_sample) / (frame + 1)`
- Reset `frame_count → 0` when camera moves (detect in `update()`)

#### Phase 6 — Chapters 9–11: Materials
Three material types encoded as an integer tag in a `Material` struct:

| Type | Tag | Parameters | Behavior |
|------|-----|-----------|----------|
| Lambertian | 0 | `albedo: vec3` | Random hemisphere scatter |
| Metal | 1 | `albedo`, `fuzz: f32` | `reflect(dir, normal) + fuzz*rand_sphere` |
| Dielectric | 2 | `ior: f32` | Snell's law + Schlick approximation |

- Iterative bounce loop (no recursion in WGSL): `for i in 0..max_bounces`
- Accumulate `attenuation` multiplicatively; add `emittance` (for light sources later)

#### Phase 7 — Chapter 12–13: Positionable Camera + Depth of Field
- Camera uniform already carries `pos/fwd/right/up` — extend `CameraController` to update these
- Lens disk sampling for DOF: sample random point on disk of `lens_radius`, offset ray origin
- `focus_dist` scales the image plane

#### Phase 8 — Chapter 14: Final Scene
- Pass the scene as a **storage buffer** (array of `Sphere` structs) instead of hardcoded values
- Build the "many random spheres" scene on the Rust side, upload once
- This also enables runtime scene editing

---

### Key wgpu API additions needed

| Need | wgpu feature |
|------|-------------|
| Compute pipeline | `device.create_compute_pipeline(...)` |
| Storage texture (R/W) | `TextureUsages::STORAGE_BINDING` + `RGBA32Float` format |
| Bind group with read+write texture | `StorageTextureAccess::ReadWrite` |
| Scene buffer | `BufferUsages::STORAGE | COPY_DST` |
| Fullscreen draw | `draw(0..3, 0..1)` with no vertex buffer |

---

### Dependency additions to `Cargo.toml`

No major new deps needed. You already have:
- `cgmath` — math
- `bytemuck` — buffer casting
- `wgpu` — GPU

Optionally add:
- `rand` — only needed if you generate the random scene on the CPU side

---

### What changes in `lib.rs`

- `State` gains: `raytracer_pipeline: RaytracerPipeline`, `accumulation_texture`, `frame_count: u32`
- `render()` becomes two-pass: compute dispatch → render pass
- `update()` resets `frame_count` if camera moved
- `resize()` recreates accumulation texture and resets frame count
- The old `vertex_buffer`/`index_buffer`/`render_pipeline` for the pentagon get removed

---

### What stays the same

- The entire `winit` event loop and `ApplicationHandler` structure
- The `Camera` and `CameraController` (with minor additions for ray direction computation)
- The `wgpu` device/queue/surface setup
