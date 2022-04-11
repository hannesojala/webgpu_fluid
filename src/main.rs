mod util;
mod vertex;

use std::{
    mem,
    num::NonZeroU32,
    time::{Duration, Instant},
};

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
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, VertexState, BufferAddress,
};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const SIMULATION_SIZE: usize = 800;
const WORK_GROUP_DIMENSIONS: u32 = 32; // x and y
const TIMESTEP_SECONDS: f32 = 1.0 / 100.;
const GAUSSS_QUAL: u32 = 50;

#[repr(C)]
struct PushContants {
    dimension: (u32, u32),
    force_pos: (i32, i32),
    force_dir: (f32, f32),
    pressed: i32,
    dt_s: f32,
    stage: u32,
}

enum FluidStage {
    Advection,
    VorticityConfinement,
    GaussSeidelIteration,
    RemoveDivergence,
    ForceInput,
}

impl From<FluidStage> for u32 {
    fn from(stage: FluidStage) -> Self {
        match stage {
            FluidStage::Advection => 0,
            FluidStage::VorticityConfinement => 1,
            FluidStage::GaussSeidelIteration => 2,
            FluidStage::RemoveDivergence => 3,
            FluidStage::ForceInput => 4,
        }
    }
}

#[derive(Debug)]
struct SimTime {
    timestep: Duration,
    current_time: Instant,
    accum_time: Duration,
}

struct App {
    surface: Surface,                       // Window surface to render to
    surface_config: SurfaceConfiguration,   // Configuration of the surface (Format, Dimensions, etc.)
    device: Device,                         // WebGPU Device
    queue: Queue,                           // Queue which executes submitted CommandBuffers
    vertex_buffer: Buffer,                  // Buffer of vertices to be drawn
    index_buffer: Buffer,                   // Buffer of indices to be drawn
    compute_bind_group: BindGroup,
    compute_pipeline: ComputePipeline,
    vel_buff: Buffer,
    temp_buff_0: Buffer,
    temp_buff_1: Buffer,
    texture_layout: ImageDataLayout,
    texture_size: wgpu::Extent3d,
    vel_texture: wgpu::Texture,
    vel_texture_bind_group: BindGroup,      // Represents texture resources used by RenderPass
    vel_render_pipeline: RenderPipeline,
    mouse_pos: (f64,f64),
    mouse_del: (f64,f64),
    mouse_down: bool,
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
                    max_push_constant_size: std::mem::size_of::<PushContants>() as u32,
                    max_compute_workgroup_size_x: WORK_GROUP_DIMENSIONS,
                    max_compute_workgroup_size_y: WORK_GROUP_DIMENSIONS,
                    max_compute_invocations_per_workgroup: WORK_GROUP_DIMENSIONS
                        * WORK_GROUP_DIMENSIONS,
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
        let velocity_buffer_bytes =
            util::to_raw(&[(0., 0., 0., 0.); SIMULATION_SIZE * SIMULATION_SIZE]);
        let vel_buff = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: velocity_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let temp_buffer_bytes = util::to_raw(&[0.0; SIMULATION_SIZE * SIMULATION_SIZE]);
        let temp_buff_0 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("temp_buff_0"),
            contents: temp_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let temp_buff_1 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("temp_buff_1"),
            contents: temp_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // Create textures
        let texture_size = wgpu::Extent3d {
            width: SIMULATION_SIZE as u32,
            height: SIMULATION_SIZE as u32,
            depth_or_array_layers: 1,
        };
        let vel_texture = device.create_texture(&TextureDescriptor {
            label: Some("velTexture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        });
        let texture_layout = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: NonZeroU32::new(mem::size_of::<(f32,f32,f32,f32)>() as u32 * texture_size.width),
            rows_per_image: NonZeroU32::new(texture_size.height),
        };
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
        // Create a shader module from an include string
        let render_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let compute_shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Input Shader"),
            source: ShaderSource::Wgsl(include_str!("fluid.wgsl").into()),
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
            label: Some("main pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("main pipeline PipelineLayoutDescriptor"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[compute_push_constant_range],
            })),
            module: &compute_shader,
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
        let vel_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &vel_texture.create_view(&wgpu::TextureViewDescriptor::default()),
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
        let vel_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            vel_buff,
            temp_buff_0,
            temp_buff_1,
            vel_texture,
            texture_size,
            texture_layout,
            vel_texture_bind_group,
            vel_render_pipeline,
            compute_bind_group,
            compute_pipeline,
            mouse_pos: (0.0,0.0),
            mouse_del: (0.0,0.0),
            mouse_down: false,
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
        render_pass.set_pipeline(&self.vel_render_pipeline);
        render_pass.set_bind_group(0, &self.vel_texture_bind_group, &[]);
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

    fn get_push_consts(&self, stage: FluidStage) -> PushContants {
        // Convert the mouse window coordinates to simulation coordinates
        let force_pos = (
            self.mouse_pos.0 as i32 * SIMULATION_SIZE as i32
                / self.surface_config.width as i32,
            self.mouse_pos.1 as i32 * SIMULATION_SIZE as i32
                / self.surface_config.height as i32,
        );
        PushContants {
            dimension: (SIMULATION_SIZE as u32, SIMULATION_SIZE as u32),
            force_pos,
            force_dir: (self.mouse_del.0 as f32, self.mouse_del.1 as f32),
            pressed: self.mouse_down as i32,
            dt_s: self.sim_time.timestep.as_secs_f32(),
            stage: stage.into(),
        }
    }

    // Update state and tell compute shader what to do
    fn update(&mut self, timesteps: u32) {
        let (dispatch_width, dispatch_height) = util::compute_work_group_count(
            (SIMULATION_SIZE as u32, SIMULATION_SIZE as u32),
            (WORK_GROUP_DIMENSIONS, WORK_GROUP_DIMENSIONS),
        );

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Compute Encoder"),});

        // Zero temp buffers
        // Comment out for buggy fake pressure (look at the compressible continuity eq for an idea why this works?)
        // todo: 
        encoder.clear_buffer(&self.temp_buff_0, 0, None);
        encoder.clear_buffer(&self.temp_buff_1, 0, None);

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute Pass"),
        });
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

        for _ in 0..timesteps {
            // Pass: Advection
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::Advection)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            // Pass: Confine Vorticity
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::VorticityConfinement)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            // Pass: Gauss-Seidel iteration
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::GaussSeidelIteration)]));
            for _ in 0..GAUSSS_QUAL {
                // Swap temp buffers
                // WHY OH WHY OH WHY I WILL NEVER DISPARAGE VULKAN SYNCHRONIZATION EVER AGAIN!
                // encoder.copy_buffer_to_buffer(&self.temp_buff_1, 0, &self.temp_buff_0, 0, self.temp_buff_size);
                compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            }
            // Pass: Remove divergence
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::RemoveDivergence)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            // Pass: Input
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::ForceInput)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
        }
        drop(compute_pass);

        // Swap simulation buffers and copy to texture to visualize
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: &self.vel_buff,
                layout: self.texture_layout,
            },
            ImageCopyTexture {
                texture: &self.vel_texture,
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
        println!(
            "Simulating {:?}ms {num_timesteps} times.",
            self.sim_time.timestep.as_millis()
        );
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
                self.mouse_del = (position.x - self.mouse_pos.0, position.y - self.mouse_pos.1);
                self.mouse_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput { button, state, .. } => match button {
                winit::event::MouseButton::Left => {
                    self.mouse_down = match state {
                        ElementState::Pressed => true,
                        ElementState::Released => false,
                    }
                }
                winit::event::MouseButton::Right => {
                    
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
        self.mouse_del = (0.0,0.0);
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(SIMULATION_SIZE as u32, SIMULATION_SIZE as u32))
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
