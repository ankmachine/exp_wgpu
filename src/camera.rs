use std::f32::consts::FRAC_PI_2;
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
// CameraController — spherical orbit model
//
// The camera always orbits around `camera.target`.  Its position is stored as
// spherical coordinates (yaw, pitch, radius) and converted to a Cartesian eye
// position every frame.
//
//   eye.x = target.x + radius * cos(pitch) * sin(yaw)
//   eye.y = target.y + radius * sin(pitch)
//   eye.z = target.z + radius * cos(pitch) * cos(yaw)
//
// Controls
// ────────
//   Left-click drag   orbit  (dx → yaw,  dy → pitch)
//   Scroll wheel      zoom   (proportional to current radius)
//   W / ↑             zoom in
//   S / ↓             zoom out
//   A / ←             orbit left   (yaw increases = CCW from above)
//   D / →             orbit right  (yaw decreases = CW  from above)
//   Space             orbit up     (pitch increases)
//   Left-Shift        orbit down   (pitch decreases)
// ---------------------------------------------------------------------------

pub struct CameraController {
    // ── Keyboard ─────────────────────────────────────────────────────────────
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,

    // ── Spherical orbit state ─────────────────────────────────────────────────
    /// Azimuth angle in radians: 0 = camera is in the +Z direction from target,
    /// π/2 = camera is in the +X direction.
    orbit_yaw: f32,
    /// Elevation angle in radians, clamped to (−π/2, π/2).
    orbit_pitch: f32,
    /// Distance from the camera origin to the look-at target.
    orbit_radius: f32,

    // ── Mouse drag ────────────────────────────────────────────────────────────
    drag_sensitivity: f32,
    zoom_sensitivity: f32,
    is_left_mouse_pressed: bool,
    last_cursor_x: f64,
    last_cursor_y: f64,
    /// False until the first CursorMoved event after a button-press; avoids a
    /// large jump on the very first drag frame.
    has_last_cursor: bool,

    // ── Dirty flag ────────────────────────────────────────────────────────────
    /// Set whenever orbit state changes; cleared (and the camera eye updated)
    /// inside `update_camera`.
    orbit_dirty: bool,
}

impl CameraController {
    /// Create a controller with the given linear keyboard step.
    /// Call [`init_from_camera`] immediately after to seed the orbit state.
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,

            orbit_yaw: 0.0,
            orbit_pitch: 0.0,
            orbit_radius: 5.0,

            drag_sensitivity: 0.005,
            zoom_sensitivity: 0.1,
            is_left_mouse_pressed: false,
            last_cursor_x: 0.0,
            last_cursor_y: 0.0,
            has_last_cursor: false,

            orbit_dirty: false,
        }
    }

    /// Seed the spherical coordinates from the current camera position so the
    /// first frame renders from exactly where the camera was placed.
    pub fn init_from_camera(&mut self, camera: &Camera) {
        use cgmath::InnerSpace;
        let offset = camera.eye - camera.target;
        let radius = offset.magnitude().max(f32::EPSILON);
        self.orbit_radius = radius;
        // asin is defined on [−1, 1]; clamp to guard against fp rounding.
        self.orbit_pitch = (offset.y / radius).clamp(-1.0, 1.0).asin();
        // atan2(x, z): angle from +Z toward +X in the XZ plane.
        self.orbit_yaw = offset.x.atan2(offset.z);
    }

    // ── Input handlers ────────────────────────────────────────────────────────

    /// Returns `true` if `key` is a recognised camera key.
    pub fn handle_key(&mut self, key: KeyCode, is_pressed: bool) -> bool {
        match key {
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

    /// Track left-mouse-button presses and releases for drag-to-orbit.
    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if button == MouseButton::Left {
            self.is_left_mouse_pressed = pressed;
            if !pressed {
                // Reset so the next press starts without a stale delta.
                self.has_last_cursor = false;
            }
        }
    }

    /// Feed the current cursor position in physical pixels.
    /// Call on every `WindowEvent::CursorMoved`.
    pub fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        if self.is_left_mouse_pressed {
            if self.has_last_cursor {
                let dx = (x - self.last_cursor_x) as f32;
                let dy = (y - self.last_cursor_y) as f32;

                // dx > 0 (cursor right) → camera orbits clockwise from above → yaw decreases.
                self.orbit_yaw -= dx * self.drag_sensitivity;

                // dy > 0 (cursor down in screen space) → camera descends → pitch decreases.
                self.orbit_pitch -= dy * self.drag_sensitivity;
                self.orbit_pitch = self.orbit_pitch.clamp(-FRAC_PI_2 + 0.02, FRAC_PI_2 - 0.02);

                self.orbit_dirty = true;
            }
            self.last_cursor_x = x;
            self.last_cursor_y = y;
            self.has_last_cursor = true;
        }
    }

    /// Feed a scroll-wheel delta.
    /// `y_delta` is positive for "scroll up" (zoom in) and negative for "scroll down" (zoom out).
    pub fn handle_scroll(&mut self, y_delta: f32) {
        // Proportional zoom: feels the same at any distance.
        self.orbit_radius *= 1.0 - y_delta * self.zoom_sensitivity;
        self.orbit_radius = self.orbit_radius.clamp(0.1, 5000.0);
        self.orbit_dirty = true;
    }

    // ── Per-frame update ──────────────────────────────────────────────────────

    /// Apply all pending inputs, write the new `camera.eye`, and return `true`
    /// if the camera moved this frame (triggering an accumulation reset).
    pub fn update_camera(&mut self, camera: &mut Camera) -> bool {
        // ── Keyboard orbit ────────────────────────────────────────────────────
        let any_key = self.is_forward_pressed
            || self.is_backward_pressed
            || self.is_left_pressed
            || self.is_right_pressed
            || self.is_up_pressed
            || self.is_down_pressed;

        if any_key {
            // W / S – zoom in / out (2% of current radius per frame so speed
            // is proportional regardless of how far the camera is).
            let zoom_step = (self.orbit_radius * 0.02).max(self.speed);
            if self.is_forward_pressed {
                self.orbit_radius = (self.orbit_radius - zoom_step).max(0.1);
            }
            if self.is_backward_pressed {
                self.orbit_radius = (self.orbit_radius + zoom_step).min(5000.0);
            }

            // A / D – orbit left / right.
            let yaw_step: f32 = 0.03; // radians per frame
            if self.is_left_pressed {
                self.orbit_yaw += yaw_step;
            }
            if self.is_right_pressed {
                self.orbit_yaw -= yaw_step;
            }

            // Space / Shift – orbit up / down.
            let pitch_step: f32 = 0.03;
            if self.is_up_pressed {
                self.orbit_pitch = (self.orbit_pitch + pitch_step).min(FRAC_PI_2 - 0.02);
            }
            if self.is_down_pressed {
                self.orbit_pitch = (self.orbit_pitch - pitch_step).max(-FRAC_PI_2 + 0.02);
            }

            self.orbit_dirty = true;
        }

        // ── Commit orbit state → eye position ─────────────────────────────────
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
