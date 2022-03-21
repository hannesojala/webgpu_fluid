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
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.tex_coord = model.tex_coord;
    return out;
}

// Fragment shader

[[group(0), binding(0)]]
var tex: texture_2d<f32>;
[[group(0), binding(1)]]
var samp: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    var out = textureSample(tex, samp, in.tex_coord);
    out.r = out.r + 0.01;
    return out;
}

 

 