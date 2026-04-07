// display.wgsl
// ------------
// Reads the accumulated ray-traced image from a RGBA32Float texture,
// applies gamma correction (sqrt ≈ gamma 2.0), and writes to the swapchain.
//
// Uses a single fullscreen triangle driven entirely by vertex_index — no vertex
// buffer is required.  The three vertices cover the full NDC square [-1,1]²:
//
//   vi=0 → (-1, -1)   (bottom-left)
//   vi=1 → ( 3, -1)   (far right,  off-screen)
//   vi=2 → (-1,  3)   (far top,    off-screen)
//

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vi], 0.0, 1.0);
    return out;
}

// ---------------------------------------------------------------------------
// Accumulation texture
// Written by the ray-tracer compute shader (Phase 2+).
// Stores *linear-light* RGBA32Float values (one sample or a running average).
//
// textureLoad is used instead of textureSample so that no sampler is needed;
// this avoids the float32-filterable feature requirement on some backends.
// ---------------------------------------------------------------------------
@group(0) @binding(0)
var acc_texture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // @builtin(position) gives fragment-centre coords (e.g. 0.5, 1.5, …).
    // floor() converts them to integer texel indices  (e.g. 0,   1,   …).
    let coord       = vec2<i32>(floor(in.clip_position.xy));
    let linear_rgb  = textureLoad(acc_texture, coord, 0).rgb;

    // Gamma correction: linear light → display (gamma ≈ 2.0 via sqrt).
    // Clamp first to guard against NaN / negative values from the accumulator.
    let display_rgb = sqrt(clamp(linear_rgb, vec3<f32>(0.0), vec3<f32>(1.0)));
    return vec4<f32>(display_rgb, 1.0);
}
