struct PushConstants {
    dimension: vec2<u32>;
    force_pos: vec2<i32>;
    force_dir: vec2<f32>;
    pressed: i32;
    dt_s: f32;
};

var<push_constant> pc: PushConstants;

struct Velocity {
    data: array<vec4<f32>>;
};

struct FloatArr {
    data: array<f32>;
};

[[group(0), binding(0)]] var<storage, read_write> velocity: Velocity;
[[group(0), binding(1)]] var<storage, read_write> temp1: FloatArr;
[[group(0), binding(2)]] var<storage, read_write> temp2: FloatArr;

fn idx(coords: vec2<u32>) -> u32 {
    return coords.x + u32(pc.dimension.x) * coords.y;
}

fn vel_at(coords: vec2<u32>) -> vec2<f32> {
    return velocity.data[idx(coords)].xy;
}

fn set_vel(coords: vec2<u32>, vel: vec2<f32>) {
    velocity.data[idx(coords)].x = vel.x;
    velocity.data[idx(coords)].y = vel.y;
}

fn blerp(v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>, v4: vec2<f32>, k: vec2<f32>) -> vec2<f32> {
    return mix(mix(v1, v2, k.x), mix(v3, v4, k.x), k.y);
}

fn bound(coords: vec2<u32>) {
    let size = vec2<u32>(pc.dimension);
    if (coords.x == 0u) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(1u, coords.y)));
    }
    if (coords.x == size.x - 1u) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(size.x - 2u, coords.y)));
    }
    if (coords.y == 0u) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(coords.x, 1u)));
    }
    if (coords.y == size.y - 1u) {
        set_vel(coords, -1.0 * vel_at(vec2<u32>(coords.x, size.y - 2u)));
    } 
}

fn add_input(coords: vec2<u32>) {
    let dt = pc.dt_s;
    let size = vec2<f32>(pc.dimension);
    let dist = distance(vec2<f32>(coords), vec2<f32>(pc.force_pos));
    let do_draw = f32(pc.pressed == 1) * f32(dist < 16.0);   // TODO: draw size push const
    let new = vel_at(coords) + (do_draw * dt * pc.force_dir);
    set_vel(coords, new);
}

fn advect(coords: vec2<u32>) {
    let size = vec2<f32>(pc.dimension);
    let dt = pc.dt_s;
    // trace back
    let pos_0 = clamp(
        vec2<f32>(coords) - (dt * vel_at(coords) * size), 
        vec2<f32>(1.5, 1.5), 
        vec2<f32>(size.x - 1.0 - 1.5, size.y - 1.0 - 1.5)
    );
    // sample
    let whole = vec2<u32>(floor(pos_0)); // floor rounding?
    let fract = fract(pos_0);
    let s0 = vel_at(whole);
    let s1 = vel_at(whole + vec2<u32>(1u, 0u));
    let s2 = vel_at(whole + vec2<u32>(0u, 1u));
    let s3 = vel_at(whole + vec2<u32>(1u, 1u));
    // blerp
    let blerped = blerp(s0,s1,s2,s3,fract);
    set_vel(coords, blerped);
}

[[stage(compute), workgroup_size(32,32)]]
fn input_main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let coords = GlobalInvocationID.xy;
    let size = (pc.dimension);
    if (coords.x > 0u && coords.x < (size.x - 1u) && coords.y > 0u && coords.y < (size.y - 1u)) {
        // add input forces
        add_input(coords);
    } else {
        bound(coords);
    }
}

[[stage(compute), workgroup_size(32,32)]]
fn advect_main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let coords = GlobalInvocationID.xy;
    let size = (pc.dimension);
    if (coords.x > 0u && coords.x < (size.x - 1u) && coords.y > 0u && coords.y < (size.y - 1u)) {
        // advect
        advect(coords);
    } else {
        bound(coords);
    }
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
    let div = get_div(coords);
    // TODO: check red black iteration to prevent data race
    // swap ptrs
    let temp = temp1.data[idx(coords)];
    temp1.data[idx(coords)] = temp2.data[idx(coords)];
    temp2.data[idx(coords)] = temp;
    // do mafs
    let u = temp2.data[idx(coords - vec2<u32>(0u, 1u))];
    let d = temp2.data[idx(coords + vec2<u32>(0u, 1u))];
    let l = temp2.data[idx(coords - vec2<u32>(1u, 0u))];
    let r = temp2.data[idx(coords + vec2<u32>(1u, 0u))];
    temp1.data[idx(coords)] = (div + u + d + l + r) / 4.0;
    // todo: bound?
    // div_sol now in temp1.data
}

[[stage(compute), workgroup_size(32,32)]]
fn gauss_it_main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let coords = GlobalInvocationID.xy;
    let size = (pc.dimension);
    if (coords.x > 0u && coords.x < (size.x - 1u) && coords.y > 0u && coords.y < (size.y - 1u)) {
        // get div, gauss seidel, and remove it
        gausss(coords);
    } else {
        bound(coords);
    }
}

fn rem_div(coords: vec2<u32>) {
    // div_sol now in temp1.data
    // // removing the divergence:
    let size = vec2<f32>(pc.dimension);
    let grad_div = 0.5 * vec2<f32>(
        temp1.data[idx(coords + vec2<u32>(1u, 0u))] - temp1.data[idx(coords - vec2<u32>(1u, 0u))],
        temp1.data[idx(coords + vec2<u32>(0u, 1u))] - temp1.data[idx(coords - vec2<u32>(0u, 1u))]
    );
    let divless = vel_at(coords) - (grad_div * size);
    set_vel(coords, divless);
}

[[stage(compute), workgroup_size(32,32)]]
fn rem_div_main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let coords = GlobalInvocationID.xy;
    let size = (pc.dimension);
    if (coords.x > 0u && coords.x < (size.x - 1u) && coords.y > 0u && coords.y < (size.y - 1u)) {
        rem_div(coords);
    } else {
        bound(coords);
    }
}

fn vel_curl_at(coords: vec2<u32>) -> f32 {
    let curl = (vel_at(coords + vec2<u32>(0u,1u)).x - vel_at(coords - vec2<u32>(0u,1u)).x) 
        - (vel_at(coords + vec2<u32>(1u,0u)).y - vel_at(coords - vec2<u32>(1u,0u)).y);
    return curl;
}

fn confine_vort(coords: vec2<u32>) {
    let dt = pc.dt_s;
    let v = 15.0;
    let vort_grad = normalize(vec2<f32>(
        abs(vel_curl_at(coords - vec2<u32>(0u,1u))) - abs(vel_curl_at(coords + vec2<u32>(0u,1u))),
        abs(vel_curl_at(coords - vec2<u32>(1u,0u))) - abs(vel_curl_at(coords + vec2<u32>(1u,0u)))
    ));
    let adjusted = vel_at(coords) + (dt * v * vort_grad * vel_curl_at(coords));
}

[[stage(compute), workgroup_size(32,32)]]
fn confine_vort_main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {
    let coords = GlobalInvocationID.xy;
    let size = (pc.dimension);
    // note smaller bounds
    if (coords.x > 1u && coords.x < (size.x - 2u) && coords.y > 1u && coords.y < (size.y - 2u)) {
        confine_vort(coords);
    } else {
        // nothing
    }
}