struct Vec4Array {
    data: array<vec4<f32>>;
};

struct F32Array {
    data: array<f32>;
};

struct PushConstant {
    draw_color: vec4<f32>;
    dimension: vec2<u32>;
    force_pos: vec2<i32>;
    force_dir: vec2<f32>;
    draw_size: f32;
    dt_s: f32;
    vort: f32;
    stage: u32;
    draw_dye: u32;
    push_vel: u32;
};

[[group(0), binding(0)]] var<storage, read_write> vel_arr: Vec4Array;
[[group(0), binding(1)]] var<storage, read_write> vel_tmp: Vec4Array;
[[group(0), binding(2)]] var<storage, read_write> dye_arr: Vec4Array;
[[group(0), binding(3)]] var<storage, read_write> dye_tmp: Vec4Array;
[[group(0), binding(4)]] var<storage, read_write> tmp_arr: F32Array;
[[group(0), binding(5)]] var<storage, read_write> tmp_tmp: F32Array;

var<push_constant> pc: PushConstant;

fn idx(coords: vec2<u32>) -> u32 {
    return coords.x + u32(pc.dimension.x) * coords.y;
}

fn tmp_at(coords: vec2<u32>) -> f32 {
    return tmp_arr.data[idx(coords)];
}

fn set_tmp(coords: vec2<u32>, val: f32) {
    tmp_arr.data[idx(coords)] = val;
}

fn tmp_at_next(coords: vec2<u32>) -> f32 {
    return tmp_tmp.data[idx(coords)];
}

fn set_tmp_next(coords: vec2<u32>, val: f32) {
    tmp_tmp.data[idx(coords)] = val;
}

fn vel_at(coords: vec2<u32>) -> vec2<f32> {
    return vel_arr.data[idx(coords)].xy;
}

fn set_vel(coords: vec2<u32>, vel: vec2<f32>) {
    vel_arr.data[idx(coords)].x = vel.x;
    vel_arr.data[idx(coords)].y = vel.y;
}

fn vel_at_next(coords: vec2<u32>) -> vec2<f32> {
    return vel_tmp.data[idx(coords)].xy;
}

fn set_vel_next(coords: vec2<u32>, vel: vec2<f32>) {
    vel_tmp.data[idx(coords)].x = vel.x;
    vel_tmp.data[idx(coords)].y = vel.y;
}

fn dye_at(coords: vec2<u32>) -> vec4<f32> {
    return dye_arr.data[idx(coords)];
}

fn set_dye(coords: vec2<u32>, dye: vec4<f32>) {
    dye_arr.data[idx(coords)] = dye;
}

fn dye_at_next(coords: vec2<u32>) -> vec4<f32> {
    return dye_tmp.data[idx(coords)];
}

fn set_dye_next(coords: vec2<u32>, dye: vec4<f32>) {
    dye_tmp.data[idx(coords)] = dye;
}

fn blerp(v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>, v4: vec2<f32>, k: vec2<f32>) -> vec2<f32> {
    return mix(mix(v1, v2, k.x), mix(v3, v4, k.x), k.y);
}
// todo: merge
fn blerp_vec4(v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>, v4: vec4<f32>, k: vec2<f32>) -> vec4<f32> {
    return mix(mix(v1, v2, k.x), mix(v3, v4, k.x), k.y);
}

// TODO: this is alllll wrong
fn bound(coords: vec2<u32>) {
    let size = vec2<u32>(pc.dimension);
    let m = size.x - 1u;
    let n = size.x - 1u;
    if (coords.x == 0u) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(1u, coords.y)));
    }
    if (coords.x == m) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(m - 1u, coords.y)));
    }
    if (coords.y == 0u) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(coords.x, 1u)));
    }
    if (coords.y == n) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(coords.x, n - 1u)));
    }
    if ((coords.x == 0u && coords.y == 0u) ||
        (coords.x == 0u && coords.y ==  n) ||
        (coords.x ==  m && coords.y == 0u) ||
        (coords.x ==  m && coords.y ==  n)) {
        set_vel(coords, vec2<f32>(0.0,0.0));
    }
}

fn add_input(coords: vec2<u32>) {
    let dt = pc.dt_s;
    let size = vec2<f32>(pc.dimension);
    let range = max(size.x, size.y) * pc.draw_size / (100.0 * 16.0);
    let dist = distance(vec2<f32>(coords), vec2<f32>(pc.force_pos));
    let do_draw = dist < range;
    if (bool(pc.push_vel) && do_draw) {
        let new_vel = vel_at(coords) + (dt * pc.force_dir); // TODO: reimpl squared dropoff brush
        set_vel(coords, new_vel);
    }
    if (bool(pc.draw_dye) && do_draw) {
        let new_dye = pc.draw_color;
        set_dye(coords, new_dye);
    }
}

fn advect(coords: vec2<u32>) {
    let size = vec2<f32>(pc.dimension);
    let dt = pc.dt_s;
    let pos_0 = clamp(
        vec2<f32>(coords) - (dt * vel_at(coords) * size), 
        vec2<f32>(1.5, 1.5), 
        vec2<f32>(size.x - 1.0 - 1.5, size.y - 1.0 - 1.5)
    );
    let whole = vec2<u32>(floor(pos_0));
    let s0 = vel_at(whole);
    let s1 = vel_at(whole + vec2<u32>(1u, 0u));
    let s2 = vel_at(whole + vec2<u32>(0u, 1u));
    let s3 = vel_at(whole + vec2<u32>(1u, 1u));
    // blerp
    let blerped = blerp(s0,s1,s2,s3,fract(pos_0));
    set_vel_next(coords, blerped);
}
// TODO: merge stages!
fn advect_dye(coords: vec2<u32>) {
    let size = vec2<f32>(pc.dimension);
    let dt = pc.dt_s;
    let pos_0 = clamp(
        vec2<f32>(coords) - (dt * vel_at(coords) * size), 
        vec2<f32>(1.5, 1.5), 
        vec2<f32>(size.x - 1.0 - 1.5, size.y - 1.0 - 1.5)
    );
    let whole = vec2<u32>(floor(pos_0));
    let s0 = dye_at(whole);
    let s1 = dye_at(whole + vec2<u32>(1u, 0u));
    let s2 = dye_at(whole + vec2<u32>(0u, 1u));
    let s3 = dye_at(whole + vec2<u32>(1u, 1u));
    let blerped = blerp_vec4(s0,s1,s2,s3,fract(pos_0));
    set_dye_next(coords, blerped);
}

fn get_div(coords: vec2<u32>) -> f32 {
    let size = vec2<f32>(pc.dimension);
    let u = vel_at(coords - vec2<u32>(0u, 1u)).y;
    let d = vel_at(coords + vec2<u32>(0u, 1u)).y;
    let l = vel_at(coords - vec2<u32>(1u, 0u)).x;
    let r = vel_at(coords + vec2<u32>(1u, 0u)).x;
    return -0.5 * ((r-l)/size.x + (d-u)/size.y);
}

fn gausss(coords: vec2<u32>) {
    // swap occurrs outside now
    let u = tmp_at(coords - vec2<u32>(0u, 1u));
    let d = tmp_at(coords + vec2<u32>(0u, 1u));
    let l = tmp_at(coords - vec2<u32>(1u, 0u));
    let r = tmp_at(coords + vec2<u32>(1u, 0u));
    let next = (get_div(coords) + u + d + l + r) / 4.0; // todo: recheck math
    set_tmp_next(coords, next);
}

fn rem_div(coords: vec2<u32>) {
    let size = vec2<f32>(pc.dimension);
    let grad_div = 0.5 * vec2<f32>(
        tmp_at(coords + vec2<u32>(1u, 0u)) - tmp_at(coords - vec2<u32>(1u, 0u)),
        tmp_at(coords + vec2<u32>(0u, 1u)) - tmp_at(coords - vec2<u32>(0u, 1u))
    );
    let divless = vel_at(coords) - (grad_div * size);
    set_vel(coords, divless);
}

fn vel_curl_at(coords: vec2<u32>) -> f32 {
    return vel_at(coords + vec2<u32>(0u,1u)).x
         - vel_at(coords - vec2<u32>(0u,1u)).x
         + vel_at(coords - vec2<u32>(1u,0u)).y
         - vel_at(coords + vec2<u32>(1u,0u)).y;
}

fn confine_vort(coords: vec2<u32>) {
    let size = (pc.dimension);
    if (coords.x > 1u && coords.x < (size.x - 2u) && coords.y > 1u && coords.y < (size.y - 2u)) {
        let dt = pc.dt_s;
        var v = pc.vort;
        let vort_grad = vec2<f32>(
            abs(vel_curl_at(coords - vec2<u32>(0u,1u))) - abs(vel_curl_at(coords + vec2<u32>(0u,1u))),
            abs(vel_curl_at(coords + vec2<u32>(1u,0u))) - abs(vel_curl_at(coords - vec2<u32>(1u,0u)))
        );
        let len = max(sqrt(vort_grad.x * vort_grad.x + vort_grad.y * vort_grad.y), 0.0001);
        let vort_grad_norm = vort_grad / len;
        let adjusted = vel_at(coords) + (dt * v * vel_curl_at(coords) * vort_grad_norm);
        set_vel_next(coords, adjusted);
    }
}

fn swap_tmp(coords: vec2<u32>) {
    set_tmp(coords, tmp_at_next(coords));
}

fn swap_vel(coords: vec2<u32>) {
    set_vel(coords, vel_at_next(coords));
}

fn swap_dye(coords: vec2<u32>) {
    set_dye(coords, dye_at_next(coords));
}

// todo: enum thing?
[[stage(compute), workgroup_size(32,32)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let coords = GlobalInvocationID.xy;
    let size = (pc.dimension);
    if (coords.x > 0u && coords.x < (size.x - 1u) && coords.y > 0u && coords.y < (size.y - 1u)) {
        switch (pc.stage) {
            case 0u: { advect(coords); }
            case 1u: { swap_vel(coords); }
            case 2u: { confine_vort(coords); }
            case 3u: { gausss(coords); }
            case 4u: { swap_tmp(coords); }
            case 5u: { rem_div(coords); }
            case 6u: { add_input(coords); }
            case 7u: { advect_dye(coords); }
            case 8u: { swap_dye(coords); }
            default: {}
        }
    }
    bound(GlobalInvocationID.xy);
}
