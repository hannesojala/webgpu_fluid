mod util;
mod vertex;

use std::{
    mem,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use image::{imageops::FilterType, EncodableLayout};
use imgui::Ui;
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
    event::{ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const SIMULATION_SIZE: usize = 1024;
const WORK_GROUP_DIMENSIONS: u32 = 32; // x and y
const TIMESTEP_SECONDS: f32 = 1.0 / 60.;
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
    Advection               = 0,
    SwapVel                 = 1,
    VorticityConfinement    = 2,
    GaussSeidelIteration    = 3,
    SwapTemp                = 4,
    RemoveDivergence        = 5,
    ForceInput              = 6,
    AdvectDye               = 7,
    SwapDye                 = 8,
}

enum RenderMode {
    Velocity, Dye
}

#[derive(Debug)]
struct SimTime {
    timestep: Duration,
    current_time: Instant,
    accum_time: Duration,
    elapsed_time: Duration,
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
    vel_tmp_buff: Buffer,
    dye_buff: Buffer,
    dye_tmp_buff: Buffer,
    temp_buff_1: Buffer,
    temp_buff_2: Buffer,
    texture_layout: ImageDataLayout,
    texture_size: wgpu::Extent3d,
    render_texture: wgpu::Texture,
    render_texture_bind_group: BindGroup,      // Represents texture resources used by RenderPass
    vel_render_pipeline: RenderPipeline,
    dye_render_pipeline: RenderPipeline,
    mouse_pos: (f64,f64),
    mouse_del: (f64,f64),
    mouse_down: bool,
    sim_time: SimTime,
    render_mode: RenderMode,
    imgui: imgui::Context,
    imgui_renderer: imgui_wgpu::Renderer,
    imgui_platform: imgui_winit_support::WinitPlatform,
}

impl App {
    pub fn new(window: &Window) -> Self {
        // Create a new instance using he Vulkan backend
        let instance = Instance::new(Backends::PRIMARY);
        // Create a surface from the window (it implements RawWindowHandle)
        let surface = unsafe { instance.create_surface(&window) };
        // Try to get an adapter to a graphics device
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })).expect("No compatible adapter found! Check drivers!");
        println!("adapter: {}", adapter.get_info().name);
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
            format: TextureFormat::Bgra8Unorm,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: PresentMode::Immediate,
        };
        surface.configure(&device, &surface_config);

        let mut imgui = imgui::Context::create();
        let mut imgui_platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        imgui_platform.attach_window(
            imgui.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );
        imgui.set_ini_filename(None);
        let hidpi_factor = window.scale_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        imgui.fonts().add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            }),
        }]);
        let imgui_renderer_config = imgui_wgpu::RendererConfig {
            texture_format: surface_config.format,
            ..Default::default()
        };
        let imgui_renderer = imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, imgui_renderer_config);

        // Buffers
        // I know theres no reason to do this for a simple screen quad but I wanted
        // to see how things were different in this API and a refresher never hurts.

        // Stores the screen quad vertices
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: util::to_raw(vertex::SCREEN_QUAD_VERTICES),
            usage: BufferUsages::VERTEX,
        });
        // Stores the screen quad indices
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: util::to_raw(vertex::SCREEN_QUAD_INDICES),
            usage: BufferUsages::INDEX,
        });
        // A buffer of float zeroes
        // 4 floats per pixel (2 velocity components + two unused)
        // Wasteful, but easy to convert to a texture, and in the future the
        // extra components will be used.
        let vec4_buffer_bytes = util::to_raw(&[(0f32, 0f32, 0f32, 0f32); SIMULATION_SIZE * SIMULATION_SIZE]);
        let vel_buff = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: vec4_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let vel_tmp_buff = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_tmp_buff"),
            contents: vec4_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let gorilla = image::open("gorilla.png").expect("No gorilla :(");
        let gorilla = gorilla.resize_exact(SIMULATION_SIZE as u32, SIMULATION_SIZE as u32, FilterType::Nearest);
        let gorilla = gorilla.into_rgba32f();
        let dye_buff = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: gorilla.as_bytes(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let dye_tmp_buff = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("dye_tmp_buff"),
            contents: vec4_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let float_buffer_bytes = util::to_raw(&[0.0; SIMULATION_SIZE * SIMULATION_SIZE]);
        let temp_buff_1 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("temp_buff_1"),
            contents: float_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let temp_buff_2 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("temp_buff_2"),
            contents: float_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        // Create textures
        let texture_size = wgpu::Extent3d {
            width: SIMULATION_SIZE as u32,
            height: SIMULATION_SIZE as u32,
            depth_or_array_layers: 1,
        };
        let render_texture = device.create_texture(&TextureDescriptor {
            label: Some("vel_texture"),
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
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
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
                    resource: vel_tmp_buff.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dye_buff.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dye_tmp_buff.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: temp_buff_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: temp_buff_2.as_entire_binding(),
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
        let render_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &render_texture.create_view(&wgpu::TextureViewDescriptor::default()),
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
                entry_point: "vel_main",
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
        let dye_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                entry_point: "dye_main",
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
            surface_config,
            device,
            queue,
            vertex_buffer,
            index_buffer,
            compute_bind_group,
            compute_pipeline,
            vel_buff,
            vel_tmp_buff,
            dye_buff,
            dye_tmp_buff,
            temp_buff_1,
            temp_buff_2,
            render_texture,
            texture_layout,
            texture_size,
            render_texture_bind_group,
            vel_render_pipeline,
            dye_render_pipeline,
            mouse_pos: (0.0,0.0),
            mouse_del: (0.0,0.0),
            mouse_down: false,
            sim_time: SimTime {
                timestep: Duration::from_secs_f32(TIMESTEP_SECONDS),
                current_time: Instant::now(),
                accum_time: Duration::ZERO,
                elapsed_time: Duration::ZERO,
            },
            render_mode: RenderMode::Dye,
            imgui, 
            imgui_renderer,
            imgui_platform,
        }
    }

    pub fn resize_surface(&mut self, new_size: PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    // Begins the render pass using the provided texture view and draws
    fn render_pass(&mut self, encoder: &mut CommandEncoder, texture_view: &TextureView, ui: Ui) {
    }

    fn render(&mut self, window: &Window) {
        // Get next swapchain texture
        let swapchain_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to obtain next swapchain image");
        //
        self.imgui_platform
            .prepare_frame(self.imgui.io_mut(), &window)
            .expect("Failed to prepare frame");
        //
        let ui = self.imgui.frame();
        {
            let window = imgui::Window::new("Hello world");
            window
                .size([300.0, 100.0], imgui::Condition::FirstUseEver)
                .build(&ui, || {
                    ui.text("Hello world!");
                    ui.text("This...is...imgui-rs on WGPU!");
                    ui.separator();
                    let mouse_pos = ui.io().mouse_pos;
                    ui.text(format!(
                        "Mouse Position: ({:.1},{:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));
                });

            let window = imgui::Window::new("Hello too");
            window
                .size([400.0, 200.0], imgui::Condition::FirstUseEver)
                .position([400.0, 200.0], imgui::Condition::FirstUseEver)
                .build(&ui, || {
                    ui.text(format!("Frametime: {:?}", self.sim_time.elapsed_time));
                });

            ui.show_demo_window(&mut true);
        }
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
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &swapchain_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        let render_pipeline = match self.render_mode {
            RenderMode::Velocity => &self.vel_render_pipeline,
            RenderMode::Dye => &self.dye_render_pipeline,
        };
        render_pass.set_pipeline(render_pipeline);
        render_pass.set_bind_group(0, &self.render_texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.draw_indexed(0..6, 0, 0..1);
        self.imgui_renderer.render(ui.render(), &self.queue, &self.device, &mut render_pass);
        drop(render_pass);
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
            stage: stage as u32,
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
        // Comment temp 1,2 out for buggy fake pressure (look at the compressible continuity eq for an idea why this works?)
        // todo: 
        encoder.clear_buffer(&self.vel_tmp_buff, 0, None); 
        encoder.clear_buffer(&self.dye_tmp_buff, 0, None); 
        encoder.clear_buffer(&self.temp_buff_1, 0, None);
        encoder.clear_buffer(&self.temp_buff_2, 0, None);

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute Pass"),
        });
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

        for _ in 0..timesteps {
            // Input
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::ForceInput)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            // Advection
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::Advection)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::SwapVel)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            // Confine Vorticity
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::VorticityConfinement)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::SwapVel)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            // Gauss-Seidel iterations
            for _ in 0..GAUSSS_QUAL {
                compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::GaussSeidelIteration)]));
                compute_pass.dispatch(dispatch_width, dispatch_height, 1);
                compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::SwapTemp)]));
                compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            }
            // Remove divergence
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::RemoveDivergence)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            // Advect Dye
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::AdvectDye)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
            compute_pass.set_push_constants(0, util::to_raw(&[self.get_push_consts(FluidStage::SwapDye)]));
            compute_pass.dispatch(dispatch_width, dispatch_height, 1);
        }
        drop(compute_pass);

        // Swap simulation buffers and copy to texture to visualize
        let buff_to_render = match self.render_mode {
            RenderMode::Velocity => &self.vel_buff,
            RenderMode::Dye => &self.dye_buff,
        };
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: buff_to_render,
                layout: self.texture_layout,
            },
            ImageCopyTexture {
                texture: &self.render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.texture_size,
        );
        self.queue.submit([encoder.finish()]);
    }

    // For some purposes in the future (unlikely for this application), device
    // events may be preferred for things like raw mouse delta or kb state.
    fn handle_window_event(&mut self, event: &WindowEvent, window: &Window) -> ControlFlow {
        match event {
            WindowEvent::CloseRequested => return ControlFlow::Exit,
            WindowEvent::KeyboardInput {input, ..} => {
                match input.virtual_keycode {
                    Some(VirtualKeyCode::Key1) => self.render_mode = RenderMode::Dye,
                    Some(VirtualKeyCode::Key2) => self.render_mode = RenderMode::Velocity,
                    _ => {}
                }
            },
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

    fn run(&mut self, window: &Window) {
        let elapsed_time = self.sim_time.current_time.elapsed();
        self.sim_time.current_time = Instant::now();
        self.sim_time.accum_time += elapsed_time;
        println!("================\nelapsed: {elapsed_time:?}");
        if self.sim_time.accum_time > self.sim_time.timestep {
            self.sim_time.accum_time -= self.sim_time.timestep;
            self.update(1);
        }
        self.render(window);
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Fluid")
        .with_inner_size(PhysicalSize::new(SIMULATION_SIZE as u32, SIMULATION_SIZE as u32))
        .build(&event_loop)
        .unwrap();
    let mut app = App::new(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                *control_flow = app.handle_window_event(&event, &window)
            }
            Event::MainEventsCleared => app.run(&window),
            _ => {}
        }
    });
}
