use std::{iter, sync::Arc, time::Instant};

mod camera;
mod raytracer;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use camera::{Camera, CameraController};

/// Compute-shader dispatches per displayed frame.
/// Increase to converge faster at the cost of frame rate.
const SAMPLES_PER_FRAME: u32 = 8;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub struct State {
    // Core wgpu handles
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,

    // --- Display pipeline ---
    // The accumulation texture is an Rgba32Float surface that the ray-tracer
    // compute shader writes to every frame.  The display pipeline reads it,
    // applies sqrt() gamma correction, and blits the result to the swapchain.
    accumulation_texture: wgpu::Texture,
    accumulation_texture_view: wgpu::TextureView,
    display_pipeline: wgpu::RenderPipeline,
    display_bind_group: wgpu::BindGroup,
    display_bind_group_layout: wgpu::BindGroupLayout,

    // --- Ray-tracer compute pipeline (Phase 2) ---
    raytracer: raytracer::RaytracerPipeline,

    // --- Progressive rendering ---
    /// Samples accumulated since the last reset.
    /// Sent to the shader as `frame_count`; reset to 0 on camera move / resize.
    frame_count: u32,
    /// Wall-clock time at application startup, used to drive animated effects.
    start_time: Instant,

    // --- Camera ---
    camera: Camera,
    camera_controller: CameraController,
}

impl State {
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Creates a fresh `Rgba32Float` texture at the given dimensions.
    ///
    /// Usages:
    ///  • `STORAGE_BINDING` – compute shader writes via `textureStore`
    ///  • `TEXTURE_BINDING` – display shader reads via `textureLoad`
    ///  • `COPY_DST`        – allows future CPU-side clears / uploads
    fn create_accumulation_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Accumulation Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Builds the display bind group that exposes the accumulation texture at
    /// binding 0 (read by the display fragment shader via `textureLoad`).
    fn create_display_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Display Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            }],
        })
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    async fn new(window: Arc<Window>) -> anyhow::Result<State> {
        let size = window.inner_size();

        // --- GPU instance, surface, adapter, device, queue ---
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        // --- Surface configuration ---
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };

        // --- Accumulation texture ---
        // The compute shader writes ray-traced output here every frame.
        // No CPU upload needed; the shader fills it from frame 0 onward.
        let (accumulation_texture, accumulation_texture_view) =
            Self::create_accumulation_texture(&device, size.width, size.height);

        // --- Final scene (RTIOW Ch. 14) ---
        // Built once on the CPU; uploaded to read-only GPU storage buffers.
        let (scene, triangles) = raytracer::build_final_scene();

        // --- Ray-tracer compute pipeline ---
        let raytracer = raytracer::RaytracerPipeline::new(
            &device,
            &queue,
            size.width,
            size.height,
            &accumulation_texture_view,
            &scene,
            &triangles,
        );

        // --- Display bind group layout ---
        // Binding 0: the accumulation texture (non-filterable f32, no sampler needed).
        let display_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Display Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        // filterable: false avoids the float32-filterable feature
                        // requirement; we use textureLoad (integer coords) instead.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                }],
            });

        let display_bind_group = Self::create_display_bind_group(
            &device,
            &display_bind_group_layout,
            &accumulation_texture_view,
        );

        // --- Display render pipeline ---
        let display_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/display.wgsl").into()),
        });

        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Display Pipeline Layout"),
                bind_group_layouts: &[&display_bind_group_layout],
                push_constant_ranges: &[],
            });

        let display_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Display Pipeline"),
            layout: Some(&display_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &display_shader,
                entry_point: Some("vs_main"),
                // No vertex buffer — fullscreen triangle positions are generated
                // from @builtin(vertex_index) inside the shader.
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &display_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // fullscreen triangle must never be culled
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // --- Camera — RTIOW Ch. 14 final-scene view ---
        // eye=(13,2,3) looking at the origin, narrow FOV, subtle aperture.
        let camera = Camera {
            eye: (13.0, 2.0, 3.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: size.width as f32 / size.height as f32,
            fovy: 20.0,
            znear: 0.1,
            zfar: 1000.0,
            lens_radius: 0.01, // subtle aperture; set to 0.0 to disable DOF
            focus_dist: 10.0,  // focal plane ~10 units out, centred on the scene
        };
        let mut camera_controller = CameraController::new(0.2);
        camera_controller.init_from_camera(&camera);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            accumulation_texture,
            accumulation_texture_view,
            display_pipeline,
            display_bind_group,
            display_bind_group_layout,
            raytracer,
            frame_count: 0,
            start_time: Instant::now(),
            camera,
            camera_controller,
        })
    }

    // -----------------------------------------------------------------------
    // Resize
    // -----------------------------------------------------------------------

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.is_surface_configured = true;

        // Recreate the accumulation texture at the new size.
        let (new_tex, new_view) = Self::create_accumulation_texture(&self.device, width, height);
        self.accumulation_texture = new_tex;
        self.accumulation_texture_view = new_view;

        // Rebuild the display bind group to point at the new texture view.
        self.display_bind_group = Self::create_display_bind_group(
            &self.device,
            &self.display_bind_group_layout,
            &self.accumulation_texture_view,
        );

        // Resize the ray-tracer: new accum buffer + updated resources bind group.
        self.raytracer
            .resize(&self.device, width, height, &self.accumulation_texture_view);

        // Reset the accumulation counter — the new accum buffer is zero-filled
        // by the driver, and frame_count = 0 makes the first trace a full overwrite.
        self.frame_count = 0;

        self.camera.aspect = width as f32 / height as f32;
    }

    // -----------------------------------------------------------------------
    // Input
    // -----------------------------------------------------------------------

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if key == KeyCode::Escape && pressed {
            event_loop.exit();
        } else if self.camera_controller.handle_key(key, pressed) {
            self.frame_count = 0;
        }
    }

    fn handle_mouse_button(&mut self, button: winit::event::MouseButton, pressed: bool) {
        self.camera_controller.handle_mouse_button(button, pressed);
    }

    fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        self.camera_controller.handle_cursor_moved(x, y, &self.camera);
    }

    fn handle_scroll(&mut self, y_delta: f32) {
        self.camera_controller.handle_scroll(y_delta);
    }

    // -----------------------------------------------------------------------
    // Update (called once per frame before render)
    // -----------------------------------------------------------------------

    fn update(&mut self) {
        if self.camera_controller.update_camera(&mut self.camera) {
            // Camera moved this frame — restart accumulation.
            self.frame_count = 0;
        }

        // Update the window title every 800 samples.
        let prev = self.frame_count;
        self.frame_count = self.frame_count.saturating_add(SAMPLES_PER_FRAME);
        if prev / 800 != self.frame_count / 800 {
            self.window.set_title(&format!(
                "RTIOW  |  {} samples  —  Alt+LMB: orbit  •  Alt+MMB: pan  •  Alt+RMB: dolly  •  scroll: zoom  •  Esc: quit",
                self.frame_count
            ));
        }
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let swapchain_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // --- Compute passes: accumulate SAMPLES_PER_FRAME samples -----------
        // Each sample needs its own submit so write_buffer is visible before
        // the next dispatch reads the updated frame_count from the uniform.
        let base = self.frame_count.saturating_sub(SAMPLES_PER_FRAME);
        let elapsed = self.start_time.elapsed().as_secs_f32();
        for i in 0..SAMPLES_PER_FRAME {
            let uniforms = raytracer::RaytracerUniforms::from_camera(
                &self.camera,
                self.config.width,
                self.config.height,
                base + i,
                elapsed,
            );
            self.raytracer.update_uniforms(&self.queue, &uniforms);

            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Compute") },
            );
            self.raytracer.dispatch(&mut enc, self.config.width, self.config.height);
            self.queue.submit(iter::once(enc.finish()));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Display Encoder"),
            });

        // --- Display pass: gamma-correct and blit to the swapchain ----------
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Display Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swapchain_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.display_pipeline);
            pass.set_bind_group(0, &self.display_bind_group, &[]);
            pass.draw(0..3, 0..1); // fullscreen triangle, no vertex buffer
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// App shell (winit ApplicationHandler)
// ---------------------------------------------------------------------------

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(proxy
                        .send_event(
                            State::new(window)
                                .await
                                .expect("Unable to create canvas!!!")
                        )
                        .is_ok())
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure if the surface is lost or outdated.
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("Unable to render: {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),

            // ── Mouse orbit ──────────────────────────────────────────────────
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => state.handle_mouse_button(button, btn_state.is_pressed()),

            WindowEvent::CursorMoved { position, .. } => {
                state.handle_cursor_moved(position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.005,
                };
                state.handle_scroll(y);
            }

            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
