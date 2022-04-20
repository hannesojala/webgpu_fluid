// Vertex shader

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coord: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coord: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    in_vert: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in_vert.position, 1.0);
    out.tex_coord = in_vert.tex_coord;
    return out;
}

// Fragment shader

[[group(0), binding(0)]] var tex: texture_2d<f32>;
[[group(0), binding(1)]] var samp: sampler;

[[stage(fragment)]]
fn vel_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let vel = textureSample(tex, samp, in.tex_coord);
    let r = abs(clamp(vel.x,  0.0, 1.0));
    let g = (abs(clamp(vel.x, -1.0, 0.0)) + abs(clamp(vel.y,  0.0, 1.0))) / 2.0;
    let b = abs(clamp(vel.y, -1.0, 0.0));
    return vec4<f32>(r,g,b, 1.0);
}

[[stage(fragment)]]
fn dye_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(tex, samp, in.tex_coord);
}

 

 