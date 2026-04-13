// ===========================================================================
// main.wgsl  –  Compute shader entry point
//
// Depends on: common.wgsl  (Uniforms u, accum, display_tex, Ray)
//             rng.wgsl     (pcg, rand, rand_in_unit_disk)
//             trace.wgsl   (trace)
//
// One invocation per pixel per frame.  Generates a single jittered,
// depth-of-field-offset ray, traces it, and blends the result into the
// running-mean accumulation buffer.
// ===========================================================================

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img = vec2<u32>(u32(u.image_size.x), u32(u.image_size.y));

    // Discard threads that overshoot the image boundary (workgroup padding).
    if gid.x >= img.x || gid.y >= img.y { return; }

    // ── Seed the RNG ──────────────────────────────────────────────────────────
    // XOR the flat pixel index with a large-prime multiple of the frame counter
    // so every frame produces an independent jitter sample for each pixel.
    let pixel_idx = gid.x + gid.y * img.x;
    var rng: u32  = pcg(pixel_idx ^ (u.frame_count * 2654435761u));

    // ── Jittered pixel UV  (RTIOW Ch. 8 antialiasing) ────────────────────────
    // Add a sub-pixel random offset so each successive frame samples a slightly
    // different point within the pixel footprint.
    let sx = (f32(gid.x) + rand(&rng)) / f32(img.x);        // [0, 1)
    let sy = 1.0 - (f32(gid.y) + rand(&rng)) / f32(img.y);  // flip Y: screen-down → world-up

    // ── Image-plane offset  (fov_scale = tan(vfov / 2)) ──────────────────────
    let aspect  = f32(img.x) / f32(img.y);
    let plane_x = (sx * 2.0 - 1.0) * aspect * u.fov_scale;
    let plane_y = (sy * 2.0 - 1.0) * u.fov_scale;

    // ── Depth-of-field ray generation  (RTIOW Ch. 13) ────────────────────────
    //
    // The focal plane lies at distance `focus_dist` from the camera origin.
    // A random offset on the lens disk displaces the ray origin, while the
    // direction still points at the same focal-plane point — this creates the
    // characteristic out-of-focus blur for objects nearer or farther than the
    // focal plane.
    //
    // When lens_radius == 0.0:
    //   disk offset → (0, 0)  →  ray_orig == camera_pos
    //   focal_pt − camera_pos == focus_dist * (fwd + plane_x*right + plane_y*up)
    //   normalize() cancels focus_dist  →  identical to the pinhole formula.
    let focal_pt = u.camera_pos
        + u.focus_dist * u.camera_fwd
        + (plane_x * u.focus_dist) * u.camera_right
        + (plane_y * u.focus_dist) * u.camera_up;

    let disk     = rand_in_unit_disk(&rng) * u.lens_radius;
    let lens_off = disk.x * u.camera_right + disk.y * u.camera_up;
    let ray_orig = u.camera_pos + lens_off;

    let ray = Ray(ray_orig, normalize(focal_pt - ray_orig));

    // ── Trace one sample ──────────────────────────────────────────────────────
    let sample = trace(ray, &rng);

    // ── Progressive accumulation  –  running mean ────────────────────────────
    //
    // new_mean = old_mean * (n / (n+1))  +  sample * (1 / (n+1))
    //          = mix(old_mean, sample, 1 / (n+1))
    //
    // When frame_count == 0, alpha == 1.0 so the stale accumulation value is
    // completely overwritten.  This makes explicit buffer clears unnecessary
    // on camera-move / resize resets — resetting frame_count to 0 is enough.
    let alpha = 1.0 / f32(u.frame_count + 1u);
    let prev  = accum[pixel_idx].rgb;
    let avg   = mix(prev, sample, alpha);

    // Persist the running mean in the accumulation buffer …
    accum[pixel_idx] = vec4<f32>(avg, 1.0);

    // … and write it to the display texture.
    // The display pass (display.wgsl) will apply sqrt() gamma correction
    // before presenting to the swapchain.
    textureStore(display_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(avg, 1.0));
}
