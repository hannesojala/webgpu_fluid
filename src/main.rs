mod util;
mod vertex;

use std::{
    mem,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use image::EncodableLayout;
use pollster::block_on;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BlendState, Buffer, BufferBindingType, BufferUsages, ColorTargetState,
    ColorWrites, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Face, Features, FragmentState, FrontFace,
    ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, IndexFormat, Instance, Limits,
    MultisampleState, PipelineLayoutDescriptor, PolygonMode, PowerPreference, PresentMode,
    PrimitiveState, PrimitiveTopology, PushConstantRange, Queue, RenderPipeline,
    RequestAdapterOptions, SamplerBindingType, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    Surface, SurfaceConfiguration, TextureDescriptor, TextureFormat, TextureSampleType,
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const SIMULATION_SIZE: usize = 1024;
const TIMESTEP_SECONDS: f32 = 0.01;

#[repr(C)]
struct PushContants {
    dimension: (u32, u32),
    force_pos: (i32, i32),
    force_dir: (f32, f32),
    pressed: i32,
    timesteps: u32,
    timestep_size_secs: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Cell(f32, f32, f32, f32);

#[derive(Debug)]
struct SimTime {
    timestep: Duration,
    current_time: Instant,
    accum_time: Duration,
}

struct App {
    surface: Surface,                     // Window surface to render to
    surface_config: SurfaceConfiguration, // Configuration of the surface (Format, Dimensions, etc.)
    device: Device,                       // WebGPU Device (GPU)
    queue: Queue,                         // Queue which executes submitted CommandBuffers
    render_pipeline: RenderPipeline, // The RenderPipeline with its bindings, buffers, and targets
    vertex_buffer: Buffer,           // Buffer of vertices to be drawn
    index_buffer: Buffer,            // Buffer of indices to be drawn
    texture_bind_group: BindGroup,   // Represents texture resources used by RenderPass
    compute_bind_group: BindGroup,
    compute_pipeline: ComputePipeline,
    vel_buff_size: u64,
    vel_buff: Buffer,
    texture_layout: ImageDataLayout,
    texture: wgpu::Texture,
    texture_size: wgpu::Extent3d,
    mouse_position: (f64, f64),
    mouse_delta: (f32, f32),
    mouse_b1down: bool,
    sim_time: SimTime,
}

impl App {
    pub fn new(window: &Window) -> Self {
        // Create a new instance using he Vulkan backend
        let instance = Instance::new(Backends::VULKAN);
        // Create a surface from the window (it implements RawWindowHandle)
        let surface = unsafe { instance.create_surface(&window) };
        // Try to get an adapter to a graphics device
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("Failed to obtain adapter: ");
        // Try to get a device and its associated queue
        let (device, queue) = block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("Device"),
                features: Features::PUSH_CONSTANTS,
                limits: Limits {
                    max_push_constant_size: 128,
                    ..Default::default()
                },
            },
            None,
        ))
        .expect("Failed to obtain device: ");

        // Surface properties
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface
                .get_preferred_format(&adapter)
                .expect("Surface is incompatible with adapter."),
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: PresentMode::Immediate,
        };
        surface.configure(&device, &surface_config);

        // Create a shader module from an include string
        let render_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Render Shader Module"),
            source: ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let velocity_compute_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Velocity Compute Module"),
            source: ShaderSource::Wgsl(include_str!("compute_velocity.wgsl").into()),
        });

        // Buffers
        // I know theres now reason to do this for a simple screen quad but I wanted
        // to see how things were different in this API and a refresher never hurts.

        // Stores the screen quad vertices
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: util::to_raw(vertex::VERTICES),
            usage: BufferUsages::VERTEX,
        });
        // Stores the screen quad indices
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: util::to_raw(vertex::INDICES),
            usage: BufferUsages::INDEX,
        });
        // A buffer of float zeroes
        // 4 floats per pixel (2 velocity components + two unused)
        // Wasteful, but easy to convert to a texture, and in the future the
        // extra components might be used.
        let velocity_buffer_bytes = util::to_raw(&[Cell(0., 0., 0., 0.); SIMULATION_SIZE * SIMULATION_SIZE]);
        let vel_buff = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: velocity_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let temp_buffer_bytes = util::to_raw(&[0.0; SIMULATION_SIZE * SIMULATION_SIZE]);
        let temp_buff_0 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: temp_buffer_bytes,
            usage: BufferUsages::STORAGE,
        });
        let temp_buff_1 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: temp_buffer_bytes,
            usage: BufferUsages::STORAGE,
        });

        // Create textures
        let texture_size = wgpu::Extent3d {
            width: SIMULATION_SIZE as u32,
            height: SIMULATION_SIZE as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("vTexture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });
        // Dummy texture data
        let gorilla = image::open("./gorilla.png")
            .unwrap()
            .resize_exact(
                texture_size.width,
                texture_size.height,
                image::imageops::FilterType::Nearest,
            )
            .into_rgba32f();
        // Enqueue a texture write
        let texture_layout = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: NonZeroU32::new(mem::size_of::<Cell>() as u32 * texture_size.width),
            rows_per_image: NonZeroU32::new(texture_size.height),
        };
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            gorilla.as_bytes(),
            texture_layout,
            texture_size,
        );
        // Create texture samplers
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let compute_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Compute bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let compute_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vel_buff.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_buff_0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: temp_buff_1.as_entire_binding(),
                },
            ],
        });
        let compute_push_constant_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushContants>() as u32,
        };
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout Descriptor"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[compute_push_constant_range],
            })),
            module: &velocity_compute_shader,
            entry_point: "main",
        });

        // Create bind group layout describing texture and sampler
        let texture_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT, // Which shader stages can see this
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        // Create bind group describing a specific texture and sampler in the layout described
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
            label: Some("bind_group"),
        });
        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            // Provide the layouts this pipeline will require
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout Descriptor"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            })),
            // Specify the shader stages
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[vertex::Vertex::buffer_layout()], // The format of the buffer
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[ColorTargetState {
                    // Describes how the color will be written
                    format: surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            primitive: PrimitiveState {
                // Describes how vertices are interpreted into triangles and filled
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        App {
            surface,
            device,
            queue,
            surface_config,
            vertex_buffer,
            index_buffer,
            vel_buff_size: velocity_buffer_bytes.len() as u64,
            vel_buff,
            texture,
            texture_size,
            texture_layout,
            texture_bind_group,
            render_pipeline,
            compute_bind_group,
            compute_pipeline,
            mouse_position: (0., 0.),
            mouse_delta: (0., 0.),
            mouse_b1down: false,
            sim_time: SimTime {
                timestep: Duration::from_secs_f32(TIMESTEP_SECONDS),
                current_time: Instant::now(),
                accum_time: Duration::ZERO,
            },
        }
    }

    pub fn resize_surface(&mut self, new_size: PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    // Begins the render pass using the provided texture view and draws
    fn render_pass(&self, encoder: &mut CommandEncoder, texture_view: &TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.draw_indexed(0..6, 0, 0..1);
    }

    fn render(&mut self) {
        // Get next swapchain texture
        let swapchain_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to obtain next swapchain image: ");
        // Create a texture view of the swapchain texture
        let swapchain_texture_view = swapchain_texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        // Create a command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        // Encode a RenderPass on that encoder that draws onto that texture view
        self.render_pass(&mut encoder, &swapchain_texture_view);
        // Submit the command buffer recieved from the encoder to the queue
        self.queue.submit([encoder.finish()]);
        // Present to screen
        swapchain_texture.present();
    }

    fn compute_pass(&self, encoder: &mut CommandEncoder, timesteps: u32) {
        // Begin a compute pass and set the pipeline and bing group
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute Pass"),
        });
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
        // Convert the mouse window coordinates to simulation coordinates
        let force_pos = (
            self.mouse_position.0 as i32 * SIMULATION_SIZE as i32
                / self.surface_config.width as i32,
            self.mouse_position.1 as i32 * SIMULATION_SIZE as i32
                / self.surface_config.height as i32,
        );
        // Setup push constants
        let push_const = PushContants {
            dimension: (SIMULATION_SIZE as u32, SIMULATION_SIZE as u32),
            force_pos,
            force_dir: self.mouse_delta,
            pressed: self.mouse_b1down as i32,
            timesteps,
            timestep_size_secs: self.sim_time.timestep.as_secs_f32(),
        };
        compute_pass.set_push_constants(0, util::to_raw(&[push_const]));
        // Dispatch with enough 16x16 workgroups to cover buffer
        let (dispatch_width, dispatch_height) = util::compute_work_group_count(
            (SIMULATION_SIZE as u32, SIMULATION_SIZE as u32),
            (16, 16),
        ); // snagged from... somewhere
        compute_pass.dispatch(dispatch_width, dispatch_height, 1);
    }

    // Update state and tell compute shader what to do
    fn update(&mut self, timesteps: u32) {
        // Create an encoder and encode the compute pass.
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        self.compute_pass(&mut encoder, timesteps);
        // Swap simulation buffers and copy to texture to visualize
        // Note on perf: 
        // These operations could be avoided if I was less lazy and stupid
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: &self.vel_buff,
                layout: self.texture_layout,
            },
            ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.texture_size,
        );
        self.queue.submit([encoder.finish()]);
    }

    // returns the amount of accumulated timesteps to simulate
    fn perf(&mut self) -> u32 {
        let elapsed_time = self.sim_time.current_time.elapsed();
        self.sim_time.current_time = Instant::now();
        self.sim_time.accum_time += elapsed_time;
        let mut num_timesteps = 0;
        println!("================\nelapsed: {elapsed_time:?}");
        println!("accumulated: {:?}", self.sim_time.accum_time);
        while self.sim_time.accum_time > self.sim_time.timestep {
            self.sim_time.accum_time -= self.sim_time.timestep;
            num_timesteps += 1;
        }
        println!("Simulating {:?}ms {num_timesteps} times.", self.sim_time.timestep.as_millis());
        num_timesteps
    }

    // For some purposes in the future (unlikely for this application), device
    // events may be preferred for things like raw mouse delta or kb state.
    fn handle_window_event(&mut self, event: &WindowEvent) -> ControlFlow {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => return ControlFlow::Exit,
            WindowEvent::Resized(size) => self.resize_surface(*size),
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_delta = (
                    (position.x - self.mouse_position.0) as f32,
                    (position.y - self.mouse_position.1) as f32,
                );
                self.mouse_position = (position.x, position.y);
            }
            WindowEvent::MouseInput { button, state, .. } => match button {
                winit::event::MouseButton::Left => {
                    self.mouse_b1down = match state {
                        ElementState::Pressed => true,
                        ElementState::Released => false,
                    }
                }
                _ => {}
            },
            _ => {}
        };
        ControlFlow::Poll
    }

    fn run(&mut self) {
        let timesteps = self.perf();
        self.update(timesteps);
        self.render();
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(720, 720))
        .build(&event_loop)
        .unwrap();
    let mut app = App::new(&window);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                *control_flow = app.handle_window_event(&event)
            }
            Event::MainEventsCleared => app.run(),
            _ => {}
        }
    });
}
