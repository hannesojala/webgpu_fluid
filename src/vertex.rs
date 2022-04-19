// Vertex Type
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    // Vertex attributes
    // TODO: Wouldn't it be cool if there was a macro to automatically create an attrib array from a struct?
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3, // Position
        1 => Float32x2, // Texture coords
    ];
    // The layout, analogous to the input binding description in Vulkan
    pub fn buffer_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress, // Number of bytes between elements
            step_mode: wgpu::VertexStepMode::Vertex, // Whether to step per vertex or per instance
            attributes: &Self::ATTRIBUTES,           // The attributes to use
        }
    }
}

// Screen quad positions
pub const SCREEN_QUAD_VERTICES: &[Vertex] = &[
    Vertex {
        position: [1.0, 1.0, 0.0],
        tex_coords: [1.0, 0.0],
    },
    Vertex {
        position: [-1., 1.0, 0.0],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [-1., -1., 0.0],
        tex_coords: [0.0, 1.0],
    },
    Vertex {
        position: [1.0, -1., 0.0],
        tex_coords: [1.0, 1.0],
    },
];

// Screen quad indices for rendering as a triangle list
pub const SCREEN_QUAD_INDICES: &[u32] = &[0, 1, 2, 0, 2, 3];
