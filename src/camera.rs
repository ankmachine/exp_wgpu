use std::f32::consts::FRAC_PI_2;
use cgmath::InnerSpace;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    /// Lens aperture radius for depth-of-field.  0.0 = perfect pinhole.
    pub lens_radius: f32,
    /// Focal-plane distance from the camera origin.
    /// 0.0 = auto (computed as ‖eye − target‖ in RaytracerUniforms::from_camera).
    pub focus_dist: f32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

impl Camera {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

// ---------------------------------------------------------------------------
// CameraUniform  (kept for potential future rasteriser use)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

// ---------------------------------------------------------------------------
// CameraController — Maya-style orbit / pan / dolly
//
// Controls
// ────────
//   Alt + LMB drag    Tumble  (orbit yaw / pitch around target)
//   Alt + MMB drag    Track   (pan target + eye together in camera plane)
//   Alt + RMB drag    Dolly   (move camera toward / away from target)
//   Scroll wheel      Dolly
//   W / ↑             Dolly in
//   S / ↓             Dolly out
//   A / ←             Orbit left
//   D / →             Orbit right
//   Space             Orbit up
//   Left-Shift        Orbit down
// ---------------------------------------------------------------------------

pub struct CameraController {
    // ── Keyboard ──────────────────────────────────────────────────────────────
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_alt_pressed: bool,

    // ── Spherical orbit state ─────────────────────────────────────────────────
    orbit_yaw: f32,
    orbit_pitch: f32,
    orbit_radius: f32,

    // ── Mouse state ───────────────────────────────────────────────────────────
    drag_sensitivity: f32,
    zoom_sensitivity: f32,
    pan_sensitivity: f32,
    is_left_mouse_pressed: bool,
    is_middle_mouse_pressed: bool,
    is_right_mouse_pressed: bool,
    last_cursor_x: f64,
    last_cursor_y: f64,
    has_last_cursor: bool,

    // ── Pending pan offset (applied to target in update_camera) ───────────────
    pan_delta: cgmath::Vector3<f32>,

    // ── Dirty flag ────────────────────────────────────────────────────────────
    orbit_dirty: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_alt_pressed: false,

            orbit_yaw: 0.0,
            orbit_pitch: 0.0,
            orbit_radius: 5.0,

            drag_sensitivity: 0.005,
            zoom_sensitivity: 0.1,
            pan_sensitivity: 0.002,
            is_left_mouse_pressed: false,
            is_middle_mouse_pressed: false,
            is_right_mouse_pressed: false,
            last_cursor_x: 0.0,
            last_cursor_y: 0.0,
            has_last_cursor: false,

            pan_delta: cgmath::Vector3::new(0.0, 0.0, 0.0),

            orbit_dirty: false,
        }
    }

    /// Seed the spherical coordinates from the current camera position.
    pub fn init_from_camera(&mut self, camera: &Camera) {
        let offset = camera.eye - camera.target;
        let radius = offset.magnitude().max(f32::EPSILON);
        self.orbit_radius = radius;
        self.orbit_pitch = (offset.y / radius).clamp(-1.0, 1.0).asin();
        self.orbit_yaw = offset.x.atan2(offset.z);
    }

    // ── Input handlers ────────────────────────────────────────────────────────

    /// Returns `true` if `key` is a recognised camera key.
    pub fn handle_key(&mut self, key: KeyCode, is_pressed: bool) -> bool {
        match key {
            KeyCode::AltLeft | KeyCode::AltRight => {
                self.is_alt_pressed = is_pressed;
                // Not a movement key itself — don't signal dirty.
                false
            }
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            KeyCode::Space => {
                self.is_up_pressed = is_pressed;
                true
            }
            KeyCode::ShiftLeft => {
                self.is_down_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => {
                self.is_left_mouse_pressed = pressed;
                if !pressed { self.has_last_cursor = false; }
            }
            MouseButton::Middle => {
                self.is_middle_mouse_pressed = pressed;
                if !pressed { self.has_last_cursor = false; }
            }
            MouseButton::Right => {
                self.is_right_mouse_pressed = pressed;
                if !pressed { self.has_last_cursor = false; }
            }
            _ => {}
        }
    }

    pub fn handle_cursor_moved(&mut self, x: f64, y: f64, camera: &Camera) {
        let any_drag = self.is_left_mouse_pressed
            || self.is_middle_mouse_pressed
            || self.is_right_mouse_pressed;

        if any_drag && self.has_last_cursor {
            let dx = (x - self.last_cursor_x) as f32;
            let dy = (y - self.last_cursor_y) as f32;

            if self.is_alt_pressed {
                if self.is_left_mouse_pressed {
                    // Alt + LMB — Tumble (orbit)
                    self.orbit_yaw   -= dx * self.drag_sensitivity;
                    self.orbit_pitch -= dy * self.drag_sensitivity;
                    self.orbit_pitch  = self.orbit_pitch.clamp(-FRAC_PI_2 + 0.02, FRAC_PI_2 - 0.02);
                    self.orbit_dirty  = true;
                } else if self.is_middle_mouse_pressed {
                    // Alt + MMB — Track (pan)
                    let fwd   = (camera.target - camera.eye).normalize();
                    let right = fwd.cross(camera.up).normalize();
                    let up    = right.cross(fwd);
                    let scale = self.orbit_radius * self.pan_sensitivity;
                    self.pan_delta += -right * dx * scale + up * dy * scale;
                    self.orbit_dirty = true;
                } else if self.is_right_mouse_pressed {
                    // Alt + RMB — Dolly (zoom along view axis)
                    let delta = (dx + dy) * self.zoom_sensitivity * 0.05;
                    self.orbit_radius *= 1.0 - delta;
                    self.orbit_radius  = self.orbit_radius.clamp(0.1, 5000.0);
                    self.orbit_dirty   = true;
                }
            }
        }

        if any_drag {
            self.last_cursor_x  = x;
            self.last_cursor_y  = y;
            self.has_last_cursor = true;
        }
    }

    pub fn handle_scroll(&mut self, y_delta: f32) {
        self.orbit_radius *= 1.0 - y_delta * self.zoom_sensitivity;
        self.orbit_radius = self.orbit_radius.clamp(0.1, 5000.0);
        self.orbit_dirty = true;
    }

    // ── Per-frame update ──────────────────────────────────────────────────────

    pub fn update_camera(&mut self, camera: &mut Camera) -> bool {
        let any_key = self.is_forward_pressed
            || self.is_backward_pressed
            || self.is_left_pressed
            || self.is_right_pressed
            || self.is_up_pressed
            || self.is_down_pressed;

        if any_key {
            let zoom_step = (self.orbit_radius * 0.02).max(self.speed);
            if self.is_forward_pressed {
                self.orbit_radius = (self.orbit_radius - zoom_step).max(0.1);
            }
            if self.is_backward_pressed {
                self.orbit_radius = (self.orbit_radius + zoom_step).min(5000.0);
            }

            let yaw_step: f32 = 0.03;
            if self.is_left_pressed  { self.orbit_yaw += yaw_step; }
            if self.is_right_pressed { self.orbit_yaw -= yaw_step; }

            let pitch_step: f32 = 0.03;
            if self.is_up_pressed {
                self.orbit_pitch = (self.orbit_pitch + pitch_step).min(FRAC_PI_2 - 0.02);
            }
            if self.is_down_pressed {
                self.orbit_pitch = (self.orbit_pitch - pitch_step).max(-FRAC_PI_2 + 0.02);
            }

            self.orbit_dirty = true;
        }

        // Apply accumulated pan to the target (eye follows automatically).
        if self.pan_delta.magnitude2() > 0.0 {
            camera.target += self.pan_delta;
            self.pan_delta  = cgmath::Vector3::new(0.0, 0.0, 0.0);
        }

        let moved = self.orbit_dirty;
        self.orbit_dirty = false;

        if moved {
            let cp = self.orbit_pitch.cos();
            let sp = self.orbit_pitch.sin();
            let sy = self.orbit_yaw.sin();
            let cy = self.orbit_yaw.cos();

            camera.eye = cgmath::Point3::new(
                camera.target.x + self.orbit_radius * cp * sy,
                camera.target.y + self.orbit_radius * sp,
                camera.target.z + self.orbit_radius * cp * cy,
            );
        }

        moved
    }
}
