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

    /// Water — dielectric refraction (IOR 1.33) with a blue-green tint on
    /// transmitted rays and a sine-wave ripple normal perturbation.
    /// `tint` is the colour applied as rays pass through.
    /// `ripple` controls wave amplitude (0.0 = flat, 0.12 = noticeable chop).
    pub fn water(centre: [f32; 3], radius: f32, tint: [f32; 3], ripple: f32) -> Self {
        Self {
            centre,
            radius,
            albedo: tint,
            fuzz: ripple.max(0.0),
            mat_type: 4,
            ior: 1.33,
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
// Minimal vec3 helpers for CPU-side geometry generation.
// ===========================================================================

fn v3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn v3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn v3_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn v3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn v3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn v3_len(a: [f32; 3]) -> f32 {
    v3_dot(a, a).sqrt()
}
fn v3_norm(a: [f32; 3]) -> Option<[f32; 3]> {
    let l = v3_len(a);
    if l < 1e-8 {
        None
    } else {
        Some(v3_scale(a, 1.0 / l))
    }
}

// ===========================================================================
// Cubic Bezier curve and tube tessellation.
//
// Renders a cubic Bezier curve as a polygonal tube by:
//   1. Sampling the curve at `segments + 1` positions.
//   2. Building a parallel-transport frame at each sample to avoid the
//      Frenet-Serret frame twist (common source of surface seam artifacts).
//   3. Placing a ring of `sides` vertices at each sample.
//   4. Connecting adjacent rings with two triangles per quad face.
//
// The resulting triangles are appended to the scene's triangle list and
// rendered with the same metal/Lambertian material pipeline as the cube.
// ===========================================================================

/// A cubic Bezier curve defined by four control points.
pub struct CubicBezier {
    /// Control points P0 … P3.
    pub p: [[f32; 3]; 4],
}

impl CubicBezier {
    /// Evaluate the curve at parameter `t ∈ [0, 1]`.
    pub fn eval(&self, t: f32) -> [f32; 3] {
        let [p0, p1, p2, p3] = self.p;
        let u = 1.0 - t;
        let uu = u * u;
        let tt = t * t;
        // B(t) = (1-t)³P0 + 3(1-t)²t P1 + 3(1-t)t² P2 + t³ P3
        let mut r = v3_scale(p0, uu * u);
        r = v3_add(r, v3_scale(p1, 3.0 * uu * t));
        r = v3_add(r, v3_scale(p2, 3.0 * u * tt));
        r = v3_add(r, v3_scale(p3, tt * t));
        r
    }

    /// First derivative B'(t), normalised.
    ///
    /// Returns `None` when the derivative is effectively zero (degenerate point,
    /// e.g. coincident consecutive control points).
    pub fn tangent(&self, t: f32) -> Option<[f32; 3]> {
        let [p0, p1, p2, p3] = self.p;
        let u = 1.0 - t;
        // B'(t) = 3[(1-t)²(P1-P0) + 2(1-t)t(P2-P1) + t²(P3-P2)]
        let d = v3_add(
            v3_add(
                v3_scale(v3_sub(p1, p0), 3.0 * u * u),
                v3_scale(v3_sub(p2, p1), 6.0 * u * t),
            ),
            v3_scale(v3_sub(p3, p2), 3.0 * t * t),
        );
        v3_norm(d)
    }
}

/// Tessellate a `CubicBezier` into a closed tube of `TriangleGpu` triangles.
///
/// # Parameters
/// * `curve`    – the Bezier curve to tessellate
/// * `radius`   – cross-section radius of the tube
/// * `segments` – number of rings along the curve (≥ 1; more = smoother)
/// * `sides`    – vertices per ring (≥ 3; more = rounder cross-section)
/// * `albedo`   – material colour
/// * `fuzz`     – metal roughness (0 = mirror, 1 = rough); use `mat_type = 0`
///               for Lambertian by calling `TriangleGpu::lambertian` directly
///
/// Returns `segments × sides × 2` triangles (plus up to 2 end-cap fans).
pub fn bezier_tube_triangles(
    curve: &CubicBezier,
    radius: f32,
    segments: usize,
    sides: usize,
    albedo: [f32; 3],
    fuzz: f32,
) -> Vec<TriangleGpu> {
    assert!(segments >= 1, "bezier_tube: segments must be >= 1");
    assert!(sides >= 3, "bezier_tube: sides must be >= 3");
    assert!(radius > 0.0, "bezier_tube: radius must be positive");

    let n = segments + 1; // number of rings
    let mut centers = Vec::with_capacity(n);
    let mut tangents = Vec::with_capacity(n);

    // ── Sample positions and tangents ───────────────────────────────────────
    for i in 0..n {
        let t = i as f32 / segments as f32;
        centers.push(curve.eval(t));
        // Fall back to a finite-difference tangent if the analytic derivative
        // is zero (can happen when consecutive control points coincide).
        let tan = curve.tangent(t).unwrap_or_else(|| {
            let eps = 1e-4_f32;
            let t0 = (t - eps).max(0.0);
            let t1 = (t + eps).min(1.0);
            v3_norm(v3_sub(curve.eval(t1), curve.eval(t0))).unwrap_or([0.0, 0.0, 1.0])
            // last resort: +Z
        });
        tangents.push(tan);
    }

    // ── Build parallel-transport frames ─────────────────────────────────────
    // Start with a normal perpendicular to T₀.  Prefer world-up (0,1,0);
    // fall back to world-right (1,0,0) when T is nearly vertical.
    let t0 = tangents[0];
    let up = if (1.0 - t0[1].abs()) > 1e-4 {
        [0.0_f32, 1.0, 0.0]
    } else {
        [1.0_f32, 0.0, 0.0]
    };
    let mut n_frame = v3_norm(v3_cross(t0, up)).unwrap_or([1.0, 0.0, 0.0]);

    let mut frames: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(n); // (N, B)
    for i in 0..n {
        let t_cur = tangents[i];
        // Re-orthogonalise N against the current tangent each step.
        n_frame =
            v3_norm(v3_sub(n_frame, v3_scale(t_cur, v3_dot(n_frame, t_cur)))).unwrap_or(n_frame);
        let b_frame = v3_cross(t_cur, n_frame); // B = T × N (outward)
        frames.push((n_frame, b_frame));

        // Parallel-transport: reflect N across the bisector of T_i and T_{i+1}.
        if i + 1 < n {
            let t_next = tangents[i + 1];
            let bisector = v3_add(t_cur, t_next);
            if let Some(bh) = v3_norm(bisector) {
                // reflect N across the plane with normal `bh`
                n_frame = v3_sub(n_frame, v3_scale(bh, 2.0 * v3_dot(n_frame, bh)));
            }
            // If bisector is near-zero (180° turn), keep the current N.
        }
    }

    // ── Generate vertex rings ────────────────────────────────────────────────
    let two_pi = std::f32::consts::TAU;
    let mut rings: Vec<Vec<[f32; 3]>> = Vec::with_capacity(n);
    for i in 0..n {
        let c = centers[i];
        let (nf, bf) = frames[i];
        let ring: Vec<[f32; 3]> = (0..sides)
            .map(|j| {
                let theta = j as f32 / sides as f32 * two_pi;
                let (s, co) = theta.sin_cos();
                v3_add(
                    c,
                    v3_add(v3_scale(nf, co * radius), v3_scale(bf, s * radius)),
                )
            })
            .collect();
        rings.push(ring);
    }

    // ── Stitch rings into triangles ──────────────────────────────────────────
    // Winding chosen so cross(e1, e2) points outward from the tube axis.
    let mut tris = Vec::with_capacity(segments * sides * 2);
    for i in 0..segments {
        for j in 0..sides {
            let j1 = (j + 1) % sides;
            // Quad: ring[i][j], ring[i][j1], ring[i+1][j1], ring[i+1][j]
            // Split into two CCW triangles (outward normal):
            tris.push(TriangleGpu::metal(
                rings[i][j],
                rings[i][j1],
                rings[i + 1][j1],
                albedo,
                fuzz,
            ));
            tris.push(TriangleGpu::metal(
                rings[i][j],
                rings[i + 1][j1],
                rings[i + 1][j],
                albedo,
                fuzz,
            ));
        }
    }

    tris
}

// ===========================================================================
// GPU triangle — packed to 80 bytes so WGSL std430 array stride matches exactly.
//
// WGSL struct layout (std430 storage):
//   offset  0  v0       vec3<f32>   (AlignOf=16 ✓)  size 12
//   offset 12  _pad0    f32                          size  4  → 16
//   offset 16  v1       vec3<f32>   (AlignOf=16 ✓)  size 12
//   offset 28  _pad1    f32                          size  4  → 32
//   offset 32  v2       vec3<f32>   (AlignOf=16 ✓)  size 12
//   offset 44  _pad2    f32                          size  4  → 48
//   offset 48  albedo   vec3<f32>   (AlignOf=16 ✓)  size 12
//   offset 60  fuzz     f32                          size  4  → 64
//   offset 64  mat_type u32                          size  4
//   offset 68  ior      f32                          size  4
//   offset 72  _pad3    f32                          size  4
//   offset 76  _pad4    f32                          size  4  → 80
//   AlignOf(struct) = 16,  SizeOf(struct) = 80 = 5 × 16 ✓
// ===========================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TriangleGpu {
    pub v0: [f32; 3],     // offset  0
    pub _pad0: f32,       // offset 12
    pub v1: [f32; 3],     // offset 16
    pub _pad1: f32,       // offset 28
    pub v2: [f32; 3],     // offset 32
    pub _pad2: f32,       // offset 44
    pub albedo: [f32; 3], // offset 48
    pub fuzz: f32,        // offset 60
    pub mat_type: u32,    // offset 64
    pub ior: f32,         // offset 68
    pub _pad3: f32,       // offset 72
    pub _pad4: f32,       // offset 76
} // total: 80 bytes

impl TriangleGpu {
    pub fn metal(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], albedo: [f32; 3], fuzz: f32) -> Self {
        Self {
            v0,
            _pad0: 0.0,
            v1,
            _pad1: 0.0,
            v2,
            _pad2: 0.0,
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0),
            mat_type: 1,
            ior: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        }
    }

    pub fn lambertian(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], albedo: [f32; 3]) -> Self {
        Self {
            v0,
            _pad0: 0.0,
            v1,
            _pad1: 0.0,
            v2,
            _pad2: 0.0,
            albedo,
            fuzz: 0.0,
            mat_type: 0,
            ior: 0.0,
            _pad3: 0.0,
            _pad4: 0.0,
        }
    }
}

/// Build the 12 triangles (6 faces × 2) of an axis-aligned cube.
///
/// `center`    – centre of the cube in world space
/// `half_size` – half-length of each edge (cube spans ±half_size on every axis)
/// `albedo`    – metal surface colour
/// `fuzz`      – metal roughness (0 = mirror, 1 = rough)
pub fn cube_triangles(
    center: [f32; 3],
    half_size: f32,
    albedo: [f32; 3],
    fuzz: f32,
) -> Vec<TriangleGpu> {
    let [cx, cy, cz] = center;
    let h = half_size;

    // 8 corners
    let v = [
        [cx - h, cy - h, cz - h], // 0: left  bottom back
        [cx + h, cy - h, cz - h], // 1: right bottom back
        [cx + h, cy + h, cz - h], // 2: right top    back
        [cx - h, cy + h, cz - h], // 3: left  top    back
        [cx - h, cy - h, cz + h], // 4: left  bottom front
        [cx + h, cy - h, cz + h], // 5: right bottom front
        [cx + h, cy + h, cz + h], // 6: right top    front
        [cx - h, cy + h, cz + h], // 7: left  top    front
    ];

    // 6 faces, each split into 2 CCW triangles (outward normals verified).
    //   Front  (+z): v4,v5,v6  v4,v6,v7
    //   Back   (-z): v1,v0,v3  v1,v3,v2
    //   Left   (-x): v0,v4,v7  v0,v7,v3
    //   Right  (+x): v5,v1,v2  v5,v2,v6
    //   Bottom (-y): v0,v1,v5  v0,v5,v4
    //   Top    (+y): v3,v7,v6  v3,v6,v2
    let faces: [[usize; 6]; 6] = [
        [4, 5, 6, 4, 6, 7], // front  +z
        [1, 0, 3, 1, 3, 2], // back   -z
        [0, 4, 7, 0, 7, 3], // left   -x
        [5, 1, 2, 5, 2, 6], // right  +x
        [0, 1, 5, 0, 5, 4], // bottom -y
        [3, 7, 6, 3, 6, 2], // top    +y
    ];

    let mut tris = Vec::with_capacity(12);
    for face in &faces {
        tris.push(TriangleGpu::metal(
            v[face[0]], v[face[1]], v[face[2]], albedo, fuzz,
        ));
        tris.push(TriangleGpu::metal(
            v[face[3]], v[face[4]], v[face[5]], albedo, fuzz,
        ));
    }
    tris
}

// ===========================================================================
// Final scene builder — RTIOW Chapter 14
//
// Produces:
//   • 1 giant ground sphere (Lambertian, grey)
//   • Up to 10×10 small random spheres (Lambertian / Metal / Dielectric)
//     filtered so they don't overlap the three showcase spheres
//   • 3 large showcase spheres  (Dielectric, Lambertian, Metal)
//   • 1 metal cube
//
// The scene is deterministic (fixed seed) so it looks the same every launch.
// ===========================================================================

pub fn build_final_scene() -> (Vec<SphereGpu>, Vec<TriangleGpu>) {
    let mut rng = SimpleRng::new(1337);
    let mut spheres = Vec::with_capacity(128);

    // --- Ground -----------------------------------------------------------
    // Note: the large central showcase sphere at (0, 1, 0) uses the fractal
    // material so it is the visual centrepiece of the scene.  Swap it back
    // to SphereGpu::dielectric([0.0, 1.0, 0.0], 1.0, 1.5) if you prefer glass.
    let ground_sphere = SphereGpu::water([0.0, -1000.0, 0.0], 1000.0, [0.05, 0.55, 0.75], 0.01);
    spheres.push(ground_sphere);

    // --- Three large showcase spheres -------------------------------------
    // Centre: fractal "dragon" Julia set — the visual centrepiece.

    spheres.push(SphereGpu::lambertian(
        [-4.0, 1.0, 0.0],
        1.0,
        [0.4, 0.2, 0.1],
    ));
    spheres.push(SphereGpu::metal([4.0, 1.0, 0.0], 1.0, [0.7, 0.6, 0.5], 0.0));

    // --- Metal cube -------------------------------------------------------
    // A polished gold-toned metal cube resting on the ground, placed to the
    // left of the scene so it reflects the showcase spheres and sky.
    let cube = cube_triangles(
        [-2.5, 0.5, 2.0], // centre (y=0.5 → bottom face sits on the ground)
        0.5,              // half-size → 1×1×1 unit cube
        [0.8, 0.6, 0.2],  // gold albedo
        0.05,             // near-mirror polish
    );

    // --- Bezier tube -------------------------------------------------------
    // A metallic blue arc that sweeps above the scene, connecting the left
    // side to the right side.  Control points chosen so the curve lifts off
    // the ground plane and returns — like a gateway arch over the spheres.
    let arc = CubicBezier {
        p: [
            [-5.0, 0.3, 1.0], // P0 – left ground anchor
            [-2.0, 4.0, 0.5], // P1 – left lift handle
            [2.0, 4.0, 0.5],  // P2 – right lift handle
            [5.0, 0.3, 1.0],  // P3 – right ground anchor
        ],
    };
    let tube = bezier_tube_triangles(
        &arc,
        0.01,            // radius
        48,              // segments along curve
        16,              // sides per ring (round cross-section)
        [0.2, 0.4, 0.9], // steel-blue albedo
        0.02,            // near-mirror polish
    );

    let mut triangles = cube;
    // render curve
    // triangles.extend(tube);

    (spheres, triangles)
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
    /// Elapsed seconds since app start — used to animate effects like water ripples.
    pub time: f32,
    pub _pad3: f32,
}

impl RaytracerUniforms {
    /// Build uniforms from the current camera state and accumulated frame count.
    pub fn from_camera(
        camera: &crate::camera::Camera,
        width: u32,
        height: u32,
        frame_count: u32,
        time: f32,
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
            time,
            _pad3: 0.0,
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

    // Group 1 – accum buffer + display texture + scene buffer + triangle buffer
    accum_buf: wgpu::Buffer,
    scene_buf: wgpu::Buffer,
    tri_buf: wgpu::Buffer,
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
    /// * `scene`     – sphere list; uploaded once, immutable for the pipeline's life.
    /// * `triangles` – triangle list; uploaded once, immutable for the pipeline's life.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        display_tex_view: &wgpu::TextureView,
        scene: &[SphereGpu],
        triangles: &[TriangleGpu],
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
            include_str!("./shaders/materials/water.wgsl"),
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
            time: 0.0,
            _pad3: 0.0,
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

        // ── Triangle buffer (group 1 / binding 3) ───────────────────────────
        let tri_data: &[u8] = if triangles.is_empty() {
            &[0u8; 80] // one dummy triangle (degenerate → never hit)
        } else {
            bytemuck::cast_slice(triangles)
        };
        let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RT Triangle Buffer"),
            contents: tri_data,
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
                    // binding 3 – triangle buffer (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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
            &tri_buf,
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
            tri_buf,
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
        tri_buf: &wgpu::Buffer,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tri_buf.as_entire_binding(),
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
            &self.tri_buf,
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
