//! Ray-tracer compute pipeline and per-frame uniform management.
//!
//! # Layout
//!
//! Two bind groups are used by the compute shader:
//!
//! | Group | Binding | Resource                        | Access      |
//! |-------|---------|---------------------------------|-------------|
//! |   0   |    0    | `RaytracerUniforms` (uniform)   | read-only   |
//! |   1   |    0    | accumulation buffer (storage)   | read + write|
//! |   1   |    1    | display texture (storage tex)   | write-only  |
//!
//! The display texture is owned by `State` (so the display render-pipeline can
//! also read it).  `RaytracerPipeline` receives a `&TextureView` at construction
//! time and rebuilds its bind group whenever the texture is recreated (resize).

use cgmath::InnerSpace;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Uniform data — must mirror the `Uniforms` struct in raytracer.wgsl exactly.
//
// Memory layout (96 bytes, six 16-byte rows):
//
//   offset  0 : camera_pos   vec3<f32>  + _pad0       f32
//   offset 16 : camera_fwd   vec3<f32>  + _pad1       f32
//   offset 32 : camera_right vec3<f32>  + fov_scale   f32
//   offset 48 : camera_up    vec3<f32>  + _pad2       f32
//   offset 64 : image_size   vec2<f32>
//   offset 72 : frame_count  u32
//   offset 76 : max_bounces  u32
//   offset 80 : lens_radius  f32
//   offset 84 : focus_dist   f32
//   offset 88 : _pad3        f32
//   offset 92 : _pad4        f32
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RaytracerUniforms {
    pub camera_pos: [f32; 3],
    pub _pad0: f32,

    pub camera_fwd: [f32; 3],
    pub _pad1: f32,

    pub camera_right: [f32; 3],
    /// `tan(vfov / 2)` — the half-height of the image plane at unit distance.
    pub fov_scale: f32,

    pub camera_up: [f32; 3],
    pub _pad2: f32,

    pub image_size: [f32; 2],
    /// Number of samples already accumulated (0 on the first frame after a reset).
    pub frame_count: u32,
    pub max_bounces: u32,

    /// Radius of the lens disk for depth-of-field.  0.0 = perfect pinhole (no blur).
    pub lens_radius: f32,
    /// Distance from the camera origin to the focal plane.
    /// When `lens_radius == 0` this value is irrelevant.
    pub focus_dist: f32,
    pub _pad3: [f32; 2],
}

impl RaytracerUniforms {
    /// Build uniforms from the current camera state and frame counter.
    ///
    /// Derives an orthonormal camera basis from `eye`, `target`, and `up`
    /// so the shader never has to do that arithmetic on every invocation.
    pub fn from_camera(
        camera: &crate::camera::Camera,
        width: u32,
        height: u32,
        frame_count: u32,
    ) -> Self {
        // Orthonormal camera basis (right-handed, −Z forward by convention).
        let fwd = (camera.target - camera.eye).normalize();
        let right = fwd.cross(camera.up).normalize();
        let up = right.cross(fwd); // re-orthogonalised up vector

        // Half-height of the image plane at distance 1 from the camera origin.
        let fov_scale = (camera.fovy.to_radians() * 0.5).tan();

        // Focus distance: explicit value, or default to the camera→target distance
        // so that the look-at point is always sharp when lens_radius == 0.
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

// ---------------------------------------------------------------------------
// RaytracerPipeline
// ---------------------------------------------------------------------------

pub struct RaytracerPipeline {
    // ---- Compute pipeline --------------------------------------------------
    pub pipeline: wgpu::ComputePipeline,

    // ---- Group 0 : uniforms ------------------------------------------------
    uniforms_buf: wgpu::Buffer,
    // uniforms_bg_layout is only needed during new() to build the pipeline
    // layout; it does not need to be retained afterward.
    uniforms_bg: wgpu::BindGroup,

    // ---- Group 1 : accum buffer + display texture --------------------------
    accum_buf: wgpu::Buffer,
    resources_bg_layout: wgpu::BindGroupLayout,
    resources_bg: wgpu::BindGroup,
}

impl RaytracerPipeline {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create the full compute pipeline.
    ///
    /// `display_tex_view` is the `TextureView` of the `Rgba32Float` accumulation
    /// texture owned by `State`.  The pipeline will write ray-traced output to it
    /// every frame; the display render-pass then reads it.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        display_tex_view: &wgpu::TextureView,
    ) -> Self {
        // ---- Shader --------------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray Tracer Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/raytracer.wgsl").into()),
        });

        // ---- Uniforms buffer (group 0 / binding 0) -------------------------
        let initial_uniforms = RaytracerUniforms {
            camera_pos: [0.0, 1.0, 2.0],
            _pad0: 0.0,
            camera_fwd: [0.0, 0.0, -1.0],
            _pad1: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            fov_scale: (45_f32.to_radians() * 0.5).tan(),
            camera_up: [0.0, 1.0, 0.0],
            _pad2: 0.0,
            image_size: [width as f32, height as f32],
            frame_count: 0,
            max_bounces: 50,
            lens_radius: 0.0,
            focus_dist: 1.0,
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

        // ---- Accumulation buffer (group 1 / binding 0) --------------------
        // Each pixel stores a vec4<f32> running mean (rgba, 16 bytes).
        // The buffer is zero-initialised by the driver (WebGPU spec guarantee),
        // so no explicit clear is needed on first use.
        let accum_buf = Self::make_accum_buf(device, width, height);

        // ---- Resources bind group layout (group 1) -------------------------
        let resources_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RT Resources BG Layout"),
                entries: &[
                    // binding 0 – accum buffer (read + write storage)
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
                    // binding 1 – display texture (write-only storage texture)
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
                ],
            });

        let resources_bg =
            Self::make_resources_bg(device, &resources_bg_layout, &accum_buf, display_tex_view);

        // ---- Compute pipeline ----------------------------------------------
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RT Pipeline Layout"),
            bind_group_layouts: &[&uniforms_bg_layout, &resources_bg_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray Tracer Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // uniforms_bg_layout is consumed into pipeline_layout above and is no
        // longer needed — drop it here rather than storing it in the struct.
        drop(uniforms_bg_layout);

        Self {
            pipeline,
            uniforms_buf,
            uniforms_bg,
            accum_buf,
            resources_bg_layout,
            resources_bg,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Allocate a zero-filled accumulation buffer sized for `width × height` pixels.
    fn make_accum_buf(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        // 4 channels (rgba) × 4 bytes per f32 = 16 bytes per pixel.
        let size = (width * height * 16) as u64;
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RT Accum Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false, // zero-initialised by the WebGPU driver
        })
    }

    /// Build the resources bind group that ties the accum buffer and display
    /// texture view together.  Called at construction and again on resize.
    fn make_resources_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        accum_buf: &wgpu::Buffer,
        display_tex_view: &wgpu::TextureView,
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
            ],
        })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Called when the window is resized.
    ///
    /// Recreates the accumulation buffer for the new pixel dimensions and
    /// rebuilds the resources bind group to point at the new display texture view.
    ///
    /// The new buffer is zero-initialised by the driver.  Combined with the
    /// `frame_count = 0` reset in `State::resize`, the progressive accumulator
    /// starts clean without any explicit memset.
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
        );
    }

    /// Upload a new set of uniforms to the GPU.
    /// Must be called once per frame (after any camera movement) before `dispatch`.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &RaytracerUniforms) {
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::cast_slice(&[*uniforms]));
    }

    /// Encode a compute dispatch into `encoder` that covers every pixel of a
    /// `width × height` image.  Each workgroup handles an 8 × 8 tile.
    ///
    /// This must be called *before* the display render-pass in the same encoder
    /// so the written display texture is visible to the subsequent fragment shader.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Ray Tracer Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.uniforms_bg, &[]);
        pass.set_bind_group(1, &self.resources_bg, &[]);

        // Ceil-divide so every pixel is covered even when dimensions aren't
        // multiples of the workgroup size (8 × 8).
        let wg_x = width.div_ceil(8);
        let wg_y = height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
