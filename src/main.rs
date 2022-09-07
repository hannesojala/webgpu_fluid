mod util;
mod vertex;

use std::{
    mem,
    num::NonZeroU32,
    path::Path,
    time::{Duration, Instant},
};

use image::{imageops::FilterType, EncodableLayout};
use pollster::block_on;
use util::to_raw;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BlendState, Buffer, BufferBindingType, BufferUsages, ColorTargetState,
    ColorWrites, CommandEncoder, CommandEncoderDescriptor, ComputePass, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, Extent3d, Face, Features,
    FragmentState, FrontFace, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, IndexFormat,
    Instance, Limits, MultisampleState, PipelineLayoutDescriptor, PolygonMode, PowerPreference,
    PresentMode, PrimitiveState, PrimitiveTopology, PushConstantRange, Queue, RenderPipeline,
    RequestAdapterOptions, SamplerBindingType, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    Surface, SurfaceConfiguration, Texture, TextureDescriptor, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{DeviceId, ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder, WindowId},
};

const WINDOW_SIZE: u32 = 800;
const SIM_SIZE: usize = 800; // TODO DYNAMICIZE
const WG_SIZE: usize = 32;

#[repr(C)]
struct FluidPushConstant {
    draw_color: [f32; 4],
    dimension: (u32, u32),
    force_pos: (i32, i32),
    force_dir: (f32, f32),
    draw_size: f32,
    dt_s: f32,
    vort: f32,
    stage: u32,
    draw_dye: u32,
    push_vel: u32,
}

enum FluidShaderStage {
    AdvectVelocity = 0,
    SwapVelocity = 1,
    VorticityConfinement = 2,
    Project = 3,
    SwapTmp = 4,
    RemoveDivergence = 5,
    Input = 6,
    AdvectDye = 7,
    SwapDye = 8,
}

struct FluidSettings {
    vort: f32,
    qual: u32,
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            vort: 5.0,
            qual: 100,
        }
    }
}

struct InputSettings {
    draw: bool,
    push: bool,
    push_power: f32,
    draw_size: f32,
    draw_color: [f32; 4],
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            push_power: 5.0,
            draw_size: 50.0,
            draw_color: [0.0, 1.0, 0.0, 1.0],
            draw: true,
            push: true,
        }
    }
}

struct InputState {
    mouse_pos: (f64, f64),
    mouse_del: (f64, f64),
    mouse_down: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            mouse_pos: (0.0, 0.0),
            mouse_del: (0.0, 0.0),
            mouse_down: false,
        }
    }
}

struct Sim {
    timestep_ms: u64, // controlled by UI, converted to Duration for use
    current_time: Instant,
    accum_time: Duration,
    elapsed_time: Duration,
}

impl Default for Sim {
    fn default() -> Self {
        Self {
            timestep_ms: 1000 / 60,
            current_time: Instant::now(),
            accum_time: Duration::ZERO,
            elapsed_time: Duration::ZERO,
        }
    }
}

struct Imgui {
    context: imgui::Context,
    renderer: imgui_wgpu::Renderer,
    winit: imgui_winit_support::WinitPlatform,
}

impl Imgui {
    fn new(window: &Window, format: TextureFormat, device: &Device, queue: &Queue) -> Self {
        let mut context = imgui::Context::create();
        let mut winit = imgui_winit_support::WinitPlatform::init(&mut context);
        winit.attach_window(
            context.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );
        context.set_ini_filename(None);
        let hidpi_factor = window.scale_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        context
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    oversample_h: 1,
                    pixel_snap_h: true,
                    size_pixels: font_size,
                    ..Default::default()
                }),
            }]);
        let renderer_config = imgui_wgpu::RendererConfig {
            texture_format: format,
            ..Default::default()
        };
        let renderer = imgui_wgpu::Renderer::new(&mut context, device, queue, renderer_config);
        Self {
            context,
            renderer,
            winit,
        }
    }

    fn handle_event<T>(&mut self, event: &Event<T>, window: &Window) {
        self.winit
            .handle_event(self.context.io_mut(), window, event);
    }
}

struct App {
    event_loop: Option<EventLoop<()>>,
    window: Window,
    wgpu: WGPU,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    compute_bind_group: BindGroup,
    compute_pipeline: ComputePipeline,
    vel_buff: Buffer,
    vel_tmp_buff: Buffer,
    dye_buff: Buffer,
    dye_tmp_buff: Buffer,
    tmp_buff_0: Buffer,
    tmp_buff_1: Buffer,
    texture_layout: ImageDataLayout,
    texture_size: Extent3d,
    render_texture: Texture,
    render_texture_bind_group: BindGroup,
    vel_render_pipeline: RenderPipeline,
    dye_render_pipeline: RenderPipeline,
    sim: Sim,
    render_mode: bool,
    imgui: Imgui,
    input_state: InputState,
    input_settings: InputSettings,
    fluid_settings: FluidSettings,
}

struct WGPU {
    surface: Surface,
    surface_config: SurfaceConfiguration,
    device: Device,
    queue: Queue,
}

impl WGPU {
    fn new(window: &Window) -> Self {
        let instance = Instance::new(Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .expect("Request adapter failed! Check drivers!");
        let (device, queue) = block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("Device"),
                features: Features::PUSH_CONSTANTS,
                limits: Limits {
                    max_push_constant_size: std::mem::size_of::<FluidPushConstant>() as u32,
                    max_compute_workgroup_size_x: WG_SIZE as u32,
                    max_compute_workgroup_size_y: WG_SIZE as u32,
                    max_compute_invocations_per_workgroup: (WG_SIZE * WG_SIZE) as u32,
                    ..Default::default()
                },
            },
            None,
        ))
        .expect("Request device failed!");
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: TextureFormat::Bgra8Unorm,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: PresentMode::Immediate,
        };
        surface.configure(&device, &surface_config);
        Self {
            surface,
            surface_config,
            device,
            queue,
        }
    }

    fn imgui(&self, window: &Window) -> Imgui {
        Imgui::new(window, self.surface_config.format, &self.device, &self.queue)
    }
}

impl App {
    pub fn new() -> Self {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Fluid")
            .with_inner_size(PhysicalSize::new(WINDOW_SIZE, WINDOW_SIZE))
            .build(&event_loop)
            .unwrap();
        let wgpu = WGPU::new(&window);
        let imgui = wgpu.imgui(&window);
        // make buffers
        // TODO move to WGPU
        let vertex_buffer = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: to_raw(vertex::SCREEN_QUAD_VERTICES),
            usage: BufferUsages::VERTEX,
        });
        let index_buffer = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: to_raw(vertex::SCREEN_QUAD_INDICES),
            usage: BufferUsages::INDEX,
        });
        let vec4_buffer_bytes = to_raw(&[(0f32, 0f32, 0f32, 0f32); SIM_SIZE * SIM_SIZE]);
        let vel_buff = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: vec4_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let vel_tmp_buff = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_tmp_buff"),
            contents: vec4_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let gorilla = image::open("gorilla.png")
            .expect("No gorilla :(")
            .resize_exact(SIM_SIZE as u32, SIM_SIZE as u32, FilterType::Nearest)
            .into_rgba32f();
        let dye_buff = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vel_buff"),
            contents: gorilla.as_bytes(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let dye_tmp_buff = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("dye_tmp_buff"),
            contents: vec4_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let float_buffer_bytes = to_raw(&[0.0; SIM_SIZE * SIM_SIZE]);
        let tmp_buff_0 = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("tmp_buff_0"),
            contents: float_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let tmp_buff_1 = wgpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("tmp_buff_1"),
            contents: float_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let texture_size = wgpu::Extent3d {
            width: SIM_SIZE as u32,
            height: SIM_SIZE as u32,
            depth_or_array_layers: 1,
        };
        let render_texture = wgpu.device.create_texture(&TextureDescriptor {
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
            bytes_per_row: NonZeroU32::new(
                mem::size_of::<(f32, f32, f32, f32)>() as u32 * texture_size.width,
            ),
            rows_per_image: NonZeroU32::new(texture_size.height),
        };
        let texture_sampler = wgpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let render_shader = wgpu.device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });
        let compute_shader = wgpu.device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("Input Shader"),
            source: ShaderSource::Wgsl(include_str!("fluid.wgsl").into()),
        });
        let compute_bind_group_layout =
            wgpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let compute_bind_group = wgpu.device.create_bind_group(&BindGroupDescriptor {
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
                    resource: tmp_buff_0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tmp_buff_1.as_entire_binding(),
                },
            ],
        });
        let compute_push_constant_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<FluidPushConstant>() as u32,
        };
        let compute_pipeline = wgpu.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("main pipeline"),
            layout: Some(&wgpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("main pipeline PipelineLayoutDescriptor"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[compute_push_constant_range],
            })),
            module: &compute_shader,
            entry_point: "main",
        });
        let texture_bind_group_layout =
            wgpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
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
        let render_texture_bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let vel_render_pipeline = wgpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&wgpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout Descriptor"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[vertex::Vertex::buffer_layout()],
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "vel_main",
                targets: &[ColorTargetState {
                    format: wgpu.surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            primitive: PrimitiveState {
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
        let dye_render_pipeline = wgpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&wgpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout Descriptor"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            })),
            vertex: VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[vertex::Vertex::buffer_layout()],
            },
            fragment: Some(FragmentState {
                module: &render_shader,
                entry_point: "dye_main",
                targets: &[ColorTargetState {
                    format: wgpu.surface_config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            primitive: PrimitiveState {
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
        // todo: impl Default
        App {
            event_loop: Some(event_loop),
            window,
            wgpu,
            vertex_buffer,
            index_buffer,
            compute_bind_group,
            compute_pipeline,
            vel_buff,
            vel_tmp_buff,
            dye_buff,
            dye_tmp_buff,
            tmp_buff_0,
            tmp_buff_1,
            texture_layout,
            texture_size,
            render_texture,
            render_texture_bind_group,
            vel_render_pipeline,
            dye_render_pipeline,
            sim: Sim::default(),
            render_mode: false,
            imgui,
            input_state: InputState::default(),
            input_settings: InputSettings::default(),
            fluid_settings: FluidSettings::default(),
        }
    }

    pub fn resize_surface(&mut self, width: u32, height: u32) {
        self.wgpu.surface_config.width = width;
        self.wgpu.surface_config.height = height;
        self.wgpu.surface.configure(&self.wgpu.device, &self.wgpu.surface_config);
    }

    fn render(&mut self) {
        let swapchain_texture = self
            .wgpu.surface
            .get_current_texture()
            .expect("Failed to obtain next swapchain image");
        self.imgui
            .winit
            .prepare_frame(self.imgui.context.io_mut(), &self.window)
            .expect("Failed to prepare frame");
        let ui = self.imgui.context.frame();
        {
            imgui::Window::new("Options (Collapsible)")
                .collapsed(true, imgui::Condition::FirstUseEver)
                .size_constraints(
                    [300.0, WINDOW_SIZE as f32],
                    [WINDOW_SIZE as f32, WINDOW_SIZE as f32],
                )
                .position([0.0, 0.0], imgui::Condition::Always)
                .build(&ui, || {
                    ui.group(|| {
                        ui.text("View Mode");
                        ui.radio_button("View Dye", &mut self.render_mode, false);
                        ui.radio_button("View Velocity", &mut self.render_mode, true);
                        ui.separator();

                        ui.text("Mouse Action");
                        ui.checkbox("Push Fluid", &mut self.input_settings.push);
                        ui.checkbox("Drop Dye", &mut self.input_settings.draw);
                        ui.separator();

                        ui.text("Input");
                        ui.set_next_item_width(-100.0);
                        imgui::Slider::new("Draw Size", 0.0, 100.0)
                            .display_format("%.2f")
                            .build(&ui, &mut self.input_settings.draw_size);
                        ui.set_next_item_width(-100.0);
                        imgui::Slider::new("Push Strength", 0.0, 10.0)
                            .display_format("%.2f")
                            .build(&ui, &mut self.input_settings.push_power);
                        imgui::ColorPicker::new("Dye Color", &mut self.input_settings.draw_color)
                            .build(&ui);
                        ui.separator();
                        ui.text_wrapped("Try dragging and dropping a png into this window!");

                        ui.text("Parameters");
                        ui.set_next_item_width(-100.0);
                        imgui::Slider::new("Vorticity", 0.0, 5.0)
                            .display_format("%.2f")
                            .build(&ui, &mut self.fluid_settings.vort);
                        ui.separator();

                        imgui::Slider::new("Quality", 5, 150)
                            .display_format("%d")
                            .build(&ui, &mut self.fluid_settings.qual);
                        imgui::Slider::new("Timestep (ms)", 1, 100)
                            .display_format("%d")
                            .build(&ui, &mut self.sim.timestep_ms);

                        ui.text_wrapped("Upcoming features: Pressure Poisson, more sliders");
                    });
                });
        }
        self.imgui.winit.prepare_render(&ui, &self.window);
        let swapchain_texture_view = swapchain_texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .wgpu.device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
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
        let render_pipeline = if self.render_mode {
            &self.vel_render_pipeline
        } else {
            &self.dye_render_pipeline
        };
        render_pass.set_pipeline(render_pipeline);
        render_pass.set_bind_group(0, &self.render_texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.draw_indexed(0..6, 0, 0..1);
        self.imgui
            .renderer
            .render(ui.render(), &self.wgpu.queue, &self.wgpu.device, &mut render_pass)
            .unwrap();
        drop(render_pass);
        self.wgpu.queue.submit([encoder.finish()]);
        swapchain_texture.present();
    }

    fn push_constant(&self, stage: FluidShaderStage) -> Vec<u8> {
        let force_pos = (
            self.input_state.mouse_pos.0 as i32 * SIM_SIZE as i32
                / self.wgpu.surface_config.width as i32,
            self.input_state.mouse_pos.1 as i32 * SIM_SIZE as i32
                / self.wgpu.surface_config.height as i32,
        );
        let del_len = (self.input_state.mouse_del.0 * self.input_state.mouse_del.0
            + self.input_state.mouse_del.1 * self.input_state.mouse_del.1)
            .sqrt();
        let force_dir = (
            self.input_settings.push_power
                * (del_len * del_len * self.input_state.mouse_del.0) as f32,
            self.input_settings.push_power
                * (del_len * del_len * self.input_state.mouse_del.1) as f32,
        );
        let over_ui = self.imgui.context.io().want_capture_mouse;
        to_raw(&[FluidPushConstant {
            draw_color: self.input_settings.draw_color,
            dimension: (SIM_SIZE as u32, SIM_SIZE as u32),
            force_pos,
            force_dir,
            draw_size: self.input_settings.draw_size,
            dt_s: Duration::from_millis(self.sim.timestep_ms).as_secs_f32(),
            vort: self.fluid_settings.vort,
            stage: stage as u32,
            draw_dye: (!over_ui && self.input_settings.draw && self.input_state.mouse_down) as u32,
            push_vel: (!over_ui && self.input_settings.push && self.input_state.mouse_down) as u32,
        }])
        .to_owned()
    }

    fn clear_buffers(&mut self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.vel_tmp_buff, 0, None);
        encoder.clear_buffer(&self.dye_tmp_buff, 0, None);
        encoder.clear_buffer(&self.tmp_buff_0, 0, None);
        encoder.clear_buffer(&self.tmp_buff_1, 0, None);
    }

    fn dispatch_stage(&self, compute_pass: &mut ComputePass, stage: FluidShaderStage) {
        let (dispatch_x, dispatch_y) =
            util::dispatch_size((SIM_SIZE, SIM_SIZE), (WG_SIZE, WG_SIZE));
        compute_pass.set_push_constants(0, &self.push_constant(stage));
        compute_pass.dispatch(dispatch_x, dispatch_y, 1);
    }

    // TODO: Seperate thread
    fn update(&mut self) {
        let mut encoder = self
            .wgpu.device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });
        self.clear_buffers(&mut encoder);
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            // FUCK theres a synchro bug somewhere. Resize to reproduce
            // todo: move loop to outer function
            while self.sim.accum_time > Duration::from_millis(self.sim.timestep_ms) {
                self.sim.accum_time -= Duration::from_millis(self.sim.timestep_ms);
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::Input);
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::AdvectVelocity);
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::SwapVelocity);
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::VorticityConfinement);
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::SwapVelocity);
                for _ in 0..self.fluid_settings.qual {
                    self.dispatch_stage(&mut compute_pass, FluidShaderStage::Project);
                    self.dispatch_stage(&mut compute_pass, FluidShaderStage::SwapTmp);
                }
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::RemoveDivergence);
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::AdvectDye);
                self.dispatch_stage(&mut compute_pass, FluidShaderStage::SwapDye);
            }
        }
        // todo: Move out of update
        let buff_to_render = if self.render_mode {
            &self.vel_buff
        } else {
            &self.dye_buff
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
        self.wgpu.queue.submit([encoder.finish()]);
    }

    fn perf_update(&mut self) {
        self.sim.elapsed_time = self.sim.current_time.elapsed();
        self.sim.current_time = Instant::now();
        self.sim.accum_time += self.sim.elapsed_time;
        self.imgui
            .context
            .io_mut()
            .update_delta_time(self.sim.elapsed_time);
    }

    fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>, _device_id: DeviceId) {
        self.input_state.mouse_del = (
            position.x - self.input_state.mouse_pos.0,
            position.y - self.input_state.mouse_pos.1,
        );
        self.input_state.mouse_pos = (position.x, position.y);
    }

    fn handle_mouse_input(
        &mut self,
        button: MouseButton,
        state: ElementState,
        _device_id: DeviceId,
    ) {
        match button {
            winit::event::MouseButton::Left => {
                self.input_state.mouse_down = state == ElementState::Pressed
            }
            _ => {}
        }
    }

    fn handle_dropped_file(&mut self, path: &Path) {
        let image = image::open(path);
        if let Ok(image) = image {
            let image = image
                .resize_exact(SIM_SIZE as u32, SIM_SIZE as u32, FilterType::Nearest)
                .into_rgba32f();
            let new_dye_buff = self.wgpu.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Dropped Image Buffer"),
                contents: image.as_bytes(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            });
            let mut encoder = self
                .wgpu.device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Dropped Image Buffer Copy Encoder"),
                });
            encoder.copy_buffer_to_buffer(
                &new_dye_buff,
                0,
                &self.dye_buff,
                0,
                image.as_bytes().len() as u64,
            );
        }
    }

    fn handle_window_event(&mut self, event: WindowEvent, window_id: WindowId) -> ControlFlow {
        if window_id == self.window.id() {
            match event {
                WindowEvent::CloseRequested => return ControlFlow::Exit,
                WindowEvent::Resized(size) => self.resize_surface(size.width, size.height),
                WindowEvent::CursorMoved {
                    position,
                    device_id,
                    ..
                } => self.handle_cursor_moved(position, device_id),
                WindowEvent::MouseInput {
                    button,
                    state,
                    device_id,
                    ..
                } => self.handle_mouse_input(button, state, device_id),
                WindowEvent::DroppedFile(pathbuf) => self.handle_dropped_file(&pathbuf),
                _ => {}
            };
        }
        ControlFlow::Poll
    }

    fn run(mut self) {
        let handler = self.event_loop.take().unwrap();
        handler.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            self.imgui.handle_event(&event, &self.window);
            match event {
                Event::WindowEvent { event, window_id } => {
                    *control_flow = self.handle_window_event(event, window_id)
                }
                Event::MainEventsCleared => {
                    self.perf_update();
                    self.update(); // move to own thread to fix gnome window drag lag
                    self.render();
                }
                _ => {}
            }
        });
    }
}

fn main() {
    let app = App::new();
    app.run();
}
