// ===========================================================================
// bvh.wgsl  –  AABB ray test + iterative BVH traversal
//
// Depends on: common.wgsl  (Ray, Hit, SphereGpu, TriangleGpu,
//                           sphere_bvh, tri_bvh, spheres, triangles)
//             geometry.wgsl (sphere_hit, triangle_hit, point_at)
//
// The BVH is a flat array of BvhNode (built on the CPU).
// Leaf nodes store a contiguous range [left, left+count) into the primitive
// buffer.  The primitives were reordered during construction so this range
// is always valid as a direct index.
//
// Traversal is iterative (WGSL has no recursion).  A local stack of u32
// node indices drives the loop; depth 32 is enough for millions of prims.
// ===========================================================================

const BVH_LEAF_BIT:  u32 = 0x80000000u;
const BVH_STACK_MAX: i32 = 32;

// ---------------------------------------------------------------------------
// Slab-method AABB ray intersection
//
// Tests the ray against three axis-aligned slab pairs and checks whether the
// entry and exit intervals overlap.  Returns true if there is a hit inside
// (t_min, t_max).
// ---------------------------------------------------------------------------
fn aabb_hit(
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    r:        Ray,
    t_min:    f32,
    t_max:    f32,
) -> bool {
    let inv = 1.0 / r.dir;
    let t0  = (aabb_min - r.origin) * inv;
    let t1  = (aabb_max - r.origin) * inv;
    let lo  = min(t0, t1);
    let hi  = max(t0, t1);
    let enter = max(max(lo.x, lo.y), max(lo.z, t_min));
    let exit  = min(min(hi.x, hi.y), min(hi.z, t_max));
    return enter <= exit;
}

// ---------------------------------------------------------------------------
// Sphere BVH traversal
// ---------------------------------------------------------------------------
fn hit_spheres_bvh(
    r:     Ray,
    t_min: f32,
    t_max: f32,
    hit:   ptr<function, Hit>,
) -> bool {
    var stack:   array<u32, 32>;
    var sp       = 0i;
    stack[sp]    = 0u;
    sp++;

    var closest  = t_max;
    var found    = false;

    while sp > 0i {
        sp--;
        let node = sphere_bvh[stack[sp]];

        if !aabb_hit(node.aabb_min, node.aabb_max, r, t_min, closest) {
            continue;
        }

        if (node.right & BVH_LEAF_BIT) != 0u {
            // Leaf — test each primitive in the range.
            let prim_start = node.left;
            let prim_count = node.right & ~BVH_LEAF_BIT;
            for (var i = 0u; i < prim_count; i++) {
                let s = spheres[prim_start + i];
                let t = sphere_hit(s.centre, s.radius, r, t_min, closest);
                if t > 0.0 {
                    closest = t;
                    found   = true;
                    let p        = point_at(r, t);
                    var outward  = (p - s.centre) / s.radius;
                    let is_front = dot(r.dir, outward) < 0.0;
                    if !is_front { outward = -outward; }
                    (*hit).t           = t;
                    (*hit).pos         = p;
                    (*hit).normal      = outward;
                    (*hit).front       = is_front;
                    (*hit).sphere_idx  = prim_start + i;
                    (*hit).is_triangle = 0u;
                    (*hit).tri_idx     = 0u;
                }
            }
        } else {
            // Internal — push both children; visit closer one last (LIFO).
            stack[sp] = node.left;  sp++;
            stack[sp] = node.right; sp++;
        }
    }
    return found;
}

// ---------------------------------------------------------------------------
// Triangle BVH traversal
// ---------------------------------------------------------------------------
fn hit_triangles_bvh(
    r:     Ray,
    t_min: f32,
    t_max: f32,
    hit:   ptr<function, Hit>,
) -> bool {
    var stack:   array<u32, 32>;
    var sp       = 0i;
    stack[sp]    = 0u;
    sp++;

    var closest  = t_max;
    var found    = false;

    // Carry through any sphere hit that already narrowed closest.
    if (*hit).t > 0.0 && (*hit).t < t_max {
        closest = (*hit).t;
    }

    while sp > 0i {
        sp--;
        let node = tri_bvh[stack[sp]];

        if !aabb_hit(node.aabb_min, node.aabb_max, r, t_min, closest) {
            continue;
        }

        if (node.right & BVH_LEAF_BIT) != 0u {
            let prim_start = node.left;
            let prim_count = node.right & ~BVH_LEAF_BIT;
            for (var i = 0u; i < prim_count; i++) {
                let tri = triangles[prim_start + i];
                let t   = triangle_hit(tri.v0, tri.v1, tri.v2, r, t_min, closest);
                if t > 0.0 {
                    closest = t;
                    found   = true;
                    let p        = point_at(r, t);
                    let e1       = tri.v1 - tri.v0;
                    let e2       = tri.v2 - tri.v0;
                    var outward  = normalize(cross(e1, e2));
                    let is_front = dot(r.dir, outward) < 0.0;
                    if !is_front { outward = -outward; }
                    (*hit).t           = t;
                    (*hit).pos         = p;
                    (*hit).normal      = outward;
                    (*hit).front       = is_front;
                    (*hit).sphere_idx  = 0u;
                    (*hit).is_triangle = 1u;
                    (*hit).tri_idx     = prim_start + i;
                }
            }
        } else {
            stack[sp] = node.left;  sp++;
            stack[sp] = node.right; sp++;
        }
    }
    return found;
}
