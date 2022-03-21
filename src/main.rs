use std::{slice, mem};

use wgpu::{Backends, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, VertexState, util::{DeviceExt, BufferInitDescriptor}, BufferSlice};
use winit::{dpi::PhysicalSize, event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};

struct App {
    surface:    Surface,
    device:     Device,
    queue:      Queue,
    config:     SurfaceConfiguration,
    size:       PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}


const VERTICES: &[Vertex] = &[
    Vertex { position: [1.0, 1.0, 0.0] },
    Vertex { position: [-1., 1.0, 0.0] },
    Vertex { position: [-1., -1., 0.0] },
    Vertex { position: [1.0, -1., 0.0] },
];

const INDICES: &[u32] = &[
    0, 1, 2, 0, 2, 3
];

impl App {
    pub fn new(window: &Window) -> Self {
        let backend = Backends::VULKAN;
        let instance = Instance::new(backend);
        let surface = unsafe { instance.create_surface(&window) };
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface)
        })).expect("Failed to obtain adapter: ");
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("Device that shouldn't suck"),
            features: Features::empty(),
            limits: Limits::default(),
        }, None)).expect("Failed to obtain device: ");
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("ShaderModule"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor { 
            label: Some("Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline Layout Descriptor"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState { // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1, // 2.
                mask: !0, // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        });
        surface.configure(&device, &config);
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor{
            label: Some("Vertex Buffer"),
            contents: slice_to_u8_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor{
            label: Some("Index Buffer"),
            contents: slice_to_u8_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        App {
            surface, device, queue, config, size, render_pipeline, vertex_buffer, index_buffer
        }
    }
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }
    pub fn render(&mut self) {
        // Get the next texture to draw to in swapchain
        let swapchain_texture = self.surface
            .get_current_texture()
            .expect("Failed to obtain next swapchain image: ");
        // Create a texture view of the output texture
        let swapchain_texture_view = swapchain_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Encode commands: A render pass
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        // Create RenderPass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    wgpu::RenderPassColorAttachment {
                        view: &swapchain_texture_view, // Texture view to use
                        resolve_target: None, // for multisampling?
                        // List of operations which will used
                        ops: wgpu::Operations {
                            // LoadOp: The op done on view at begining of the pass
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1, g: 0.9, b: 0.3, a: 1.0,
                            }),
                            store: true, // Whether data is written
                        },
                    }
                ],
                depth_stencil_attachment: None, // Depth attachment, if used
            });
            // Set pipeline to use
            render_pass.set_pipeline(&self.render_pipeline);
            // Draw Command: Vertices and instances
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
        // submit iterator of command buffers and present
        self.queue.submit(std::iter::once(encoder.finish()));
        swapchain_texture.present();
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).expect("Window creation error: ");
    let mut app = App::new(&window);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape), ..
                    }, ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(size) => app.resize(*size),
                _ => {}
            },
            Event::MainEventsCleared => {
                app.render();
            },
            _ => {}
        }
    });
}

// A bad idea
fn slice_to_u8_slice<'a, T>(s: &'a [T]) -> &'a [u8] {
    unsafe {
        slice::from_raw_parts(s.as_ptr() as *const u8, mem::size_of::<T>()*s.len())
    }
}