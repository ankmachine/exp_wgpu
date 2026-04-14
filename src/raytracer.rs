//! Ray-tracer compute pipeline, scene data, and per-frame uniform management.
//!
//! # Bind-group layout
//!
//! | Group | Binding | Resource                        | Access       |
//! |-------|---------|---------------------------------|--------------|
//! |   0   |    0    | `RaytracerUniforms` (uniform)   | read-only    |
//! |   1   |    0    | accumulation buffer (storage)   | read + write |
//! |   1   |    1    | display texture (storage tex)   | write-only   |
//! |   1   |    2    | scene sphere buffer (storage)   | read-only    |
//!
//! The display texture (`Rgba32Float`) is owned by `State` so the display
//! render-pipeline can also sample it.  `RaytracerPipeline` receives a
//! `&TextureView` at construction time and rebuilds its bind group on resize.
//! The scene sphere buffer is immutable for the lifetime of the pipeline.

use cgmath::InnerSpace;
use wgpu::util::DeviceExt;

// ===========================================================================
// GPU sphere — packed to 48 bytes so WGSL std430 array stride matches exactly.
//
// WGSL struct layout (std430 storage):
//   offset  0  centre   vec3<f32>   (AlignOf=16 ✓ first member)  size 12
//   offset 12  radius   f32                                       size  4  → 16
//   offset 16  albedo   vec3<f32>   (AlignOf=16 ✓)               size 12
//   offset 28  fuzz     f32                                       size  4  → 32
//   offset 32  mat_type u32                                       size  4
//   offset 36  ior      f32                                       size  4
//   offset 40  _pad0    f32                                       size  4
//   offset 44  _pad1    f32                                       size  4  → 48
//   AlignOf(struct) = 16,  SizeOf(struct) = 48 = 3 × 16 ✓
//
// Rust #[repr(C)] with all f32/u32 fields (alignment 4) produces the same
// byte offsets because every field boundary above is a multiple of 4.
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereGpu {
    pub centre: [f32; 3], // offset  0
    pub radius: f32,      // offset 12
    pub albedo: [f32; 3], // offset 16  (surface colour for Lambertian / Metal)
    pub fuzz: f32,        // offset 28  (Metal roughness 0=mirror … 1=rough)
    pub mat_type: u32,    // offset 32  (0=Lambertian, 1=Metal, 2=Dielectric)
    pub ior: f32,         // offset 36  (index of refraction, Dielectric only)
    pub _pad0: f32,       // offset 40
    pub _pad1: f32,       // offset 44
} // total: 48 bytes

impl SphereGpu {
    pub fn lambertian(centre: [f32; 3], radius: f32, albedo: [f32; 3]) -> Self {
        Self {
            centre,
            radius,
            albedo,
            fuzz: 0.0,
            mat_type: 0,
            ior: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    pub fn metal(centre: [f32; 3], radius: f32, albedo: [f32; 3], fuzz: f32) -> Self {
        Self {
            centre,
            radius,
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
            mat_type: 1,
            ior: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    pub fn dielectric(centre: [f32; 3], radius: f32, ior: f32) -> Self {
        Self {
            centre,
            radius,
            albedo: [1.0, 1.0, 1.0],
            fuzz: 0.0,
            mat_type: 2,
            ior,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    /// A Julia-set fractal surface.
    ///
    /// The Julia set `z_{n+1} = z² + c` is evaluated on the sphere's UV surface
    /// and the smooth escape time is mapped to a colour palette.  Rays scatter
    /// diffusely (Lambertian) with the fractal colour as albedo.
    ///
    /// # Field mapping (repurposed from standard SphereGpu fields)
    ///
    /// | SphereGpu field | Fractal meaning                              |
    /// |-----------------|----------------------------------------------|
    /// | `albedo.rg`     | Julia constant `c` (real, imaginary)         |
    /// | `albedo.b`      | Colour palette: 0=Rainbow 1=Magma 2=Ice 3=Gold |
    /// | `fuzz`          | UV zoom / scale (try 1.5 – 3.0)             |
    /// | `ior`           | Max iterations as f32 (try 32.0 – 128.0)    |
    ///
    /// # Interesting Julia constants
    /// * `[-0.70,  0.27]` — "dragon"  (intricate connected tendrils)
    /// * `[ 0.28,  0.01]` — "tree"    (branching structure)
    /// * `[-0.40,  0.60]` — "dust"    (disconnected island clusters)
    /// * `[-0.54,  0.54]` — "spiral"  (tight swirling arms)
    /// * `[-0.12, -0.77]` — "sun"     (radial symmetry)
    pub fn fractal(
        centre: [f32; 3],
        radius: f32,
        c: [f32; 2],    // Julia constant (real, imaginary)
        palette: f32,   // 0=Rainbow, 1=Magma, 2=Ice, 3=Gold
        zoom: f32,      // UV scale (try 2.0)
        max_iters: f32, // iteration depth (try 64.0)
    ) -> Self {
        Self {
            centre,
            radius,
            albedo: [c[0], c[1], palette],
            fuzz: zoom,
            mat_type: 3,
            ior: max_iters,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

// ===========================================================================
// Minimal deterministic PRNG for CPU-side scene generation.
// Uses a Knuth multiplicative LCG — fast and dependency-free.
// ===========================================================================

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // Avalanche the seed so consecutive small integers produce very
        // different sequences.
        let s = seed ^ 0x9e3779b97f4a7c15;
        Self(if s == 0 { 1 } else { s })
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Knuth "The Art of Computer Programming".
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform float in [0, 1).
    fn f32(&mut self) -> f32 {
        // Take the top 24 bits for 24-bit float precision.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform float in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }
}

// ===========================================================================
// Final scene builder — RTIOW Chapter 14
//
// Produces:
//   • 1 giant ground sphere (Lambertian, grey)
//   • Up to 22×22 = 484 small random spheres (Lambertian / Metal / Dielectric)
//     filtered so they don't overlap the three showcase spheres
//   • 3 large showcase spheres  (Dielectric, Lambertian, Metal)
//
// The scene is deterministic (fixed seed) so it looks the same every launch.
// ===========================================================================

pub fn build_final_scene() -> Vec<SphereGpu> {
    let mut rng = SimpleRng::new(1337);
    let mut spheres = Vec::with_capacity(512);

    // --- Ground -----------------------------------------------------------
    // Note: the large central showcase sphere at (0, 1, 0) uses the fractal
    // material so it is the visual centrepiece of the scene.  Swap it back
    // to SphereGpu::dielectric([0.0, 1.0, 0.0], 1.0, 1.5) if you prefer glass.
    spheres.push(SphereGpu::lambertian(
        [0.0, -1000.0, 0.0],
        1000.0,
        [0.5, 0.5, 0.5],
    ));

    // --- Random small spheres (22 × 22 grid) ------------------------------
    for a in -11i32..11 {
        for b in -11i32..11 {
            let cx = a as f32 + 0.9 * rng.f32();
            let cy = 0.2_f32;
            let cz = b as f32 + 0.9 * rng.f32();

            // Skip spheres that would overlap the three large showcase spheres.
            // The large spheres sit at (0,1,0), (-4,1,0), (4,1,0) with r=1.
            // A small sphere (r=0.2) at (cx,0.2,cz) is too close when the
            // centre-to-centre distance is less than 1.2.  We use 1.3 for a
            // comfortable gap.
            let too_close = |ox: f32, oz: f32| {
                let dx = cx - ox;
                let dz = cz - oz;
                (dx * dx + dz * dz).sqrt() < 1.3
            };
            if too_close(0.0, 0.0) || too_close(-4.0, 0.0) || too_close(4.0, 0.0) {
                continue;
            }

            let choose_mat = rng.f32();

            if choose_mat < 0.8 {
                // Lambertian – random saturated colour (product of two randoms
                // biases toward darker values, matching RTIOW's formula).
                let r = rng.f32() * rng.f32();
                let g = rng.f32() * rng.f32();
                let b = rng.f32() * rng.f32();
                spheres.push(SphereGpu::lambertian([cx, cy, cz], 0.2, [r, g, b]));
            } else if choose_mat < 0.95 {
                // Metal – pale colour, low-to-medium fuzz.
                let r = rng.range(0.5, 1.0);
                let g = rng.range(0.5, 1.0);
                let b = rng.range(0.5, 1.0);
                let fuzz = rng.range(0.0, 0.5);
                spheres.push(SphereGpu::metal([cx, cy, cz], 0.2, [r, g, b], fuzz));
            } else {
                // Dielectric – glass.
                spheres.push(SphereGpu::dielectric([cx, cy, cz], 0.2, 1.5));
            }
        }
    }

    // --- Three large showcase spheres -------------------------------------
    // Centre: fractal "dragon" Julia set — the visual centrepiece.
    spheres.push(SphereGpu::fractal(
        [0.0, 1.0, 0.0],
        1.0,
        [-0.70, 0.27], // Julia "dragon" constant
        0.0,           // Rainbow palette
        2.0,           // UV zoom
        64.0,          // iteration depth
    ));
    spheres.push(SphereGpu::lambertian(
        [-4.0, 1.0, 0.0],
        1.0,
        [0.4, 0.2, 0.1],
    ));
    spheres.push(SphereGpu::metal([4.0, 1.0, 0.0], 1.0, [0.7, 0.6, 0.5], 0.0));

    spheres
}

// ===========================================================================
// Uniform data — must mirror the `Uniforms` struct in raytracer.wgsl exactly.
//
// Memory layout (96 bytes, six 16-byte rows):
//
//   offset  0  camera_pos   vec3<f32>  + _pad0       f32
//   offset 16  camera_fwd   vec3<f32>  + _pad1       f32
//   offset 32  camera_right vec3<f32>  + fov_scale   f32
//   offset 48  camera_up    vec3<f32>  + _pad2       f32
//   offset 64  image_size   vec2<f32>
//   offset 72  frame_count  u32
//   offset 76  max_bounces  u32
//   offset 80  lens_radius  f32
//   offset 84  focus_dist   f32
//   offset 88  _pad3        f32
//   offset 92  _pad4        f32
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RaytracerUniforms {
    pub camera_pos: [f32; 3],
    pub _pad0: f32,

    pub camera_fwd: [f32; 3],
    pub _pad1: f32,

    pub camera_right: [f32; 3],
    /// `tan(vfov / 2)` — half-height of the image plane at unit distance.
    pub fov_scale: f32,

    pub camera_up: [f32; 3],
    pub _pad2: f32,

    pub image_size: [f32; 2],
    /// Samples accumulated since last reset (0 on first frame).
    pub frame_count: u32,
    pub max_bounces: u32,

    /// Lens aperture radius.  0.0 = perfect pinhole (no depth-of-field blur).
    pub lens_radius: f32,
    /// Distance from the camera origin to the focal plane.
    pub focus_dist: f32,
    pub _pad3: [f32; 2],
}

impl RaytracerUniforms {
    /// Build uniforms from the current camera state and accumulated frame count.
    pub fn from_camera(
        camera: &crate::camera::Camera,
        width: u32,
        height: u32,
        frame_count: u32,
    ) -> Self {
        let fwd = (camera.target - camera.eye).normalize();
        let right = fwd.cross(camera.up).normalize();
        let up = right.cross(fwd); // re-orthogonalised

        let fov_scale = (camera.fovy.to_radians() * 0.5).tan();

        // Default: focus on the look-at point so it is always sharp.
        let focus_dist = if camera.focus_dist > 0.0 {
            camera.focus_dist
        } else {
            (camera.target - camera.eye).magnitude()
        };

        Self {
            camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z],
            _pad0: 0.0,
            camera_fwd: [fwd.x, fwd.y, fwd.z],
            _pad1: 0.0,
            camera_right: [right.x, right.y, right.z],
            fov_scale,
            camera_up: [up.x, up.y, up.z],
            _pad2: 0.0,
            image_size: [width as f32, height as f32],
            frame_count,
            max_bounces: 50,
            lens_radius: camera.lens_radius,
            focus_dist,
            _pad3: [0.0; 2],
        }
    }
}

// ===========================================================================
// RaytracerPipeline
// ===========================================================================

pub struct RaytracerPipeline {
    pub pipeline: wgpu::ComputePipeline,

    // Group 0 – uniforms
    uniforms_buf: wgpu::Buffer,
    uniforms_bg: wgpu::BindGroup,

    // Group 1 – accum buffer + display texture + scene buffer
    accum_buf: wgpu::Buffer,
    scene_buf: wgpu::Buffer,
    resources_bg_layout: wgpu::BindGroupLayout,
    resources_bg: wgpu::BindGroup,
}

impl RaytracerPipeline {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build the full compute pipeline.
    ///
    /// * `display_tex_view` – view of the `Rgba32Float` texture owned by `State`
    ///   that the compute shader writes to and the display pass reads from.
    /// * `scene` – the list of spheres for the initial scene; uploaded once and
    ///   immutable for the life of this pipeline instance.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        display_tex_view: &wgpu::TextureView,
        scene: &[SphereGpu],
    ) -> Self {
        // ── Shader ──────────────────────────────────────────────────────────
        // The compute shader is assembled from focused single-responsibility files
        // concatenated at compile time.  Dependency order (each file may only use
        // symbols declared in earlier files):
        //
        //   common.wgsl              shared types + all resource bindings
        //   rng.wgsl                 PCG hash + distribution samplers
        //   geometry.wgsl            sphere_hit, hit_world
        //   sky.wgsl                 sky_color
        //   materials/lambertian     scatter_lambertian
        //   materials/metal          scatter_metal
        //   materials/dielectric     schlick, scatter_dielectric
        //   materials/fractal        fractal_palette, julia_smooth, scatter_fractal
        //   materials/dispatch       MAT_* constants + scatter() router
        //   trace.wgsl               trace() iterative path tracer
        //   main.wgsl                cs_main entry point
        //
        // To add a new material:
        //   1. Create src/shaders/materials/<name>.wgsl
        //   2. Add a case to materials/dispatch.wgsl
        //   3. Add its include_str!() line below (before dispatch.wgsl)
        let shader_source = concat!(
            include_str!("./shaders/common.wgsl"),
            include_str!("./shaders/rng.wgsl"),
            include_str!("./shaders/geometry.wgsl"),
            include_str!("./shaders/sky.wgsl"),
            include_str!("./shaders/materials/lambertian.wgsl"),
            include_str!("./shaders/materials/metal.wgsl"),
            include_str!("./shaders/materials/dielectric.wgsl"),
            include_str!("./shaders/materials/fractal.wgsl"),
            include_str!("./shaders/materials/dispatch.wgsl"),
            include_str!("./shaders/trace.wgsl"),
            include_str!("./shaders/main.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RT Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // ── Uniforms buffer (group 0 / binding 0) ───────────────────────────
        let initial_uniforms = RaytracerUniforms {
            camera_pos: [13.0, 2.0, 3.0],
            _pad0: 0.0,
            camera_fwd: [-0.919, -0.153, -0.362], // normalised (13,2,3)→(0,0,0)
            _pad1: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            fov_scale: (20_f32.to_radians() * 0.5).tan(),
            camera_up: [0.0, 1.0, 0.0],
            _pad2: 0.0,
            image_size: [width as f32, height as f32],
            frame_count: 0,
            max_bounces: 50,
            lens_radius: 0.0,
            focus_dist: 10.0,
            _pad3: [0.0; 2],
        };

        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RT Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[initial_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RT Uniforms BG Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniforms_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RT Uniforms BG"),
            layout: &uniforms_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buf.as_entire_binding(),
            }],
        });

        // ── Accumulation buffer (group 1 / binding 0) ───────────────────────
        // 4 f32 channels × 4 bytes = 16 bytes per pixel; zero-init by driver.
        let accum_buf = Self::make_accum_buf(device, width, height);

        // ── Scene buffer (group 1 / binding 2) ──────────────────────────────
        // Uploaded once; the shader iterates over it for every ray intersection.
        // If no spheres are provided we still need a non-empty buffer so
        // Metal/Vulkan validation doesn't complain about a zero-size binding.
        let scene_data: &[u8] = if scene.is_empty() {
            &[0u8; 48] // one dummy sphere (radius 0 → never hit)
        } else {
            bytemuck::cast_slice(scene)
        };
        let scene_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RT Scene Buffer"),
            contents: scene_data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Resources bind group layout (group 1) ───────────────────────────
        let resources_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RT Resources BG Layout"),
                entries: &[
                    // binding 0 – accum buffer (read + write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1 – display texture (write-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // binding 2 – scene sphere buffer (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let resources_bg = Self::make_resources_bg(
            device,
            &resources_bg_layout,
            &accum_buf,
            display_tex_view,
            &scene_buf,
        );

        // ── Compute pipeline ─────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RT Pipeline Layout"),
            bind_group_layouts: &[&uniforms_bg_layout, &resources_bg_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RT Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        drop(uniforms_bg_layout); // consumed by pipeline_layout above

        Self {
            pipeline,
            uniforms_buf,
            uniforms_bg,
            accum_buf,
            scene_buf,
            resources_bg_layout,
            resources_bg,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn make_accum_buf(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let size = (width * height * 16) as u64; // vec4<f32> per pixel
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RT Accum Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false, // WebGPU spec guarantees zero-init
        })
    }

    fn make_resources_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        accum_buf: &wgpu::Buffer,
        display_tex_view: &wgpu::TextureView,
        scene_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RT Resources BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(display_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene_buf.as_entire_binding(),
                },
            ],
        })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Recreate the accumulation buffer for the new pixel dimensions and rebuild
    /// the resources bind group pointing at the new display texture view.
    /// The scene buffer is immutable and is reused as-is.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        display_tex_view: &wgpu::TextureView,
    ) {
        self.accum_buf = Self::make_accum_buf(device, width, height);
        self.resources_bg = Self::make_resources_bg(
            device,
            &self.resources_bg_layout,
            &self.accum_buf,
            display_tex_view,
            &self.scene_buf,
        );
    }

    /// Upload updated camera uniforms to the GPU once per frame.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &RaytracerUniforms) {
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::cast_slice(&[*uniforms]));
    }

    /// Encode a compute dispatch covering every pixel of a `width × height` image.
    /// Must be called **before** the display render-pass in the same encoder.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RT Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniforms_bg, &[]);
        pass.set_bind_group(1, &self.resources_bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
}
