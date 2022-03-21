use std::{slice, mem};

use wgpu::{Backends, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, VertexState, util::{DeviceExt, BufferInitDescriptor}, BufferSlice, Texture, TextureDescriptor, TextureUsages};
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
    texture:    Texture,
    texture_view: wgpu::TextureView,
    texture_sampler: wgpu::Sampler,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    texture_size: wgpu::Extent3d,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}


const VERTICES: &[Vertex] = &[
    Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 0.0] },
    Vertex { position: [-1., 1.0, 0.0], tex_coords: [0.0, 0.0] },
    Vertex { position: [-1., -1., 0.0], tex_coords: [0.0, 1.0] },
    Vertex { position: [1.0, -1., 0.0], tex_coords: [1.0, 1.0] },
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("ShaderModule"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
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
        let texture_size = wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 };
        let texture_format = surface.get_preferred_format(&adapter).unwrap();
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });
        let mut texture_data = vec![255_u8; (config.width*4*config.height) as usize]; // white
        // Every 4th column black
        texture_data.chunks_mut(4).step_by(4).for_each(|p| { p[0] = 0; p[1] = 0; p[2] = 0; p[3] = 0; } );
        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            &texture_data,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * config.width),
                rows_per_image: std::num::NonZeroU32::new(config.height),
            },
            texture_size,
        );
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            // SamplerBindingType::Comparison is only for TextureSampleType::Depth
                            // SamplerBindingType::Filtering if the sample_type of the texture is:
                            //     TextureSampleType::Float { filterable: true }
                            // Otherwise you'll get an error.
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            }
        );        
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler),
                    }
                ],
                label: Some("bind_group"),
            }
        );
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor { 
            label: Some("Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline Layout Descriptor"),
                bind_group_layouts: &[&texture_bind_group_layout],
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
        App {
            surface, device, queue, config, size, 
            render_pipeline, vertex_buffer, index_buffer, 
            texture, texture_view, texture_sampler, texture_bind_group_layout, bind_group, texture_size
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
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            // Draw Command: Vertices and instances
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
        encoder.copy_texture_to_texture(
            swapchain_texture.texture.as_image_copy(), 
            self.texture.as_image_copy(), 
            self.texture_size
        );
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