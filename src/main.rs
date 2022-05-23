mod util;
mod vertex;

use std::{
    mem,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use image::{imageops::FilterType, EncodableLayout};
use pollster::block_on;
use util::to_raw;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BlendState, Buffer, BufferBindingType, BufferUsages, ColorTargetState,
    ColorWrites, CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Extent3d, Face, Features, FragmentState,
    FrontFace, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, IndexFormat, Instance, Limits,
    MultisampleState, PipelineLayoutDescriptor, PolygonMode, PowerPreference, PresentMode,
    PrimitiveState, PrimitiveTopology, PushConstantRange, Queue, RenderPipeline,
    RequestAdapterOptions, SamplerBindingType, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    Surface, SurfaceConfiguration, Texture, TextureDescriptor, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::{
    dpi::{PhysicalSize},
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

// TODO: MAKE DYNAMIC & CHANGE-ABLE IN UI
const WINDOW_SIZE: u32 = 800;
const SIM_SIZE: usize = 800;
const WG_SIZE: usize = 32;
const DT_S: f32 = 1.0 / 100.;
const QUAL: u32 = 50;

#[repr(C)]
struct PushConst {
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

enum Stage {
    AdvectVel = 0,
    SwapVel = 1,
    VortConf = 2,
    Project = 3,
    SwapTmp = 4,
    RemDiv = 5,
    Input = 6,
    AdvectDye = 7,
    SwapDye = 8,
}

struct App {
    surface: Surface,
    surface_config: SurfaceConfiguration,
    device: Device,
    queue: Queue,
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
    timestep: Duration,
    current_time: Instant,
    accum_time: Duration,
    elapsed_time: Duration,
    render_mode: bool,
    imgui_context: imgui::Context,
    imgui_renderer: imgui_wgpu::Renderer,
    imgui_platform: imgui_winit_support::WinitPlatform,
    mouse_pos: (f64, f64),
    mouse_del: (f64, f64),
    mouse_down: bool,
    push_power: f32,
    draw_size: f32,
    draw_color: [f32;4],
    draw_dye: bool,
    push_vel: bool,
    vort: f32,
}

impl App {
    pub fn new(window: &Window) -> Self {
        let instance = Instance::new(Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(&window) };
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
                    max_push_constant_size: std::mem::size_of::<PushConst>() as u32,
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
        imgui
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
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
        let imgui_renderer =
            imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, imgui_renderer_config);
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: to_raw(vertex::SCREEN_QUAD_VERTICES),
            usage: BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: to_raw(vertex::SCREEN_QUAD_INDICES),
            usage: BufferUsages::INDEX,
        });
        let vec4_buffer_bytes = to_raw(&[(0f32, 0f32, 0f32, 0f32); SIM_SIZE * SIM_SIZE]);
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
        let gorilla = gorilla.resize_exact(SIM_SIZE as u32, SIM_SIZE as u32, FilterType::Nearest);
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
        let float_buffer_bytes = to_raw(&[0.0; SIM_SIZE * SIM_SIZE]);
        let tmp_buff_0 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("tmp_buff_0"),
            contents: float_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let tmp_buff_1 = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("tmp_buff_1"),
            contents: float_buffer_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let texture_size = wgpu::Extent3d {
            width: SIM_SIZE as u32,
            height: SIM_SIZE as u32,
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
            bytes_per_row: NonZeroU32::new(
                mem::size_of::<(f32, f32, f32, f32)>() as u32 * texture_size.width,
            ),
            rows_per_image: NonZeroU32::new(texture_size.height),
        };
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
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
            range: 0..std::mem::size_of::<PushConst>() as u32,
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
        let texture_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let vel_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
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
                    format: surface_config.format,
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
        let dye_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
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
                    format: surface_config.format,
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
            tmp_buff_0,
            tmp_buff_1,
            texture_layout,
            texture_size,
            render_texture,
            render_texture_bind_group,
            vel_render_pipeline,
            dye_render_pipeline,
            timestep: Duration::from_secs_f32(DT_S),
            current_time: Instant::now(),
            accum_time: Duration::ZERO,
            elapsed_time: Duration::ZERO,
            render_mode: false,
            imgui_context: imgui,
            imgui_renderer,
            imgui_platform,
            mouse_pos: (0.0, 0.0),
            mouse_del: (0.0, 0.0),
            mouse_down: false,
            push_power: 5.0,
            draw_size: 50.0,
            draw_color: [0.,0.,0.,0.,],
            draw_dye: true,
            push_vel: true,
            vort: 5.0,
        }
    }

    pub fn resize_surface(&mut self, new_size: PhysicalSize<u32>) {
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn render(&mut self, window: &Window) {
        let swapchain_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to obtain next swapchain image");
        self.imgui_platform
            .prepare_frame(self.imgui_context.io_mut(), window)
            .expect("Failed to prepare frame");
        let ui = self.imgui_context.frame();
        {
            imgui::Window::new("Options (Collapsible)")
            .collapsed(true, imgui::Condition::FirstUseEver)
            .size_constraints([300.0, WINDOW_SIZE as f32], [WINDOW_SIZE as f32,WINDOW_SIZE as f32])
            .position([0.0, 0.0], imgui::Condition::Always)
            .build(&ui, ||{
                ui.group(||{
                    ui.text("View Mode");
                    ui.radio_button("View Dye", &mut self.render_mode, false);
                    ui.radio_button("View Velocity", &mut self.render_mode, true);
                    ui.separator();

                    ui.text("Mouse Action");
                    ui.checkbox("Push Fluid", &mut self.push_vel);
                    ui.checkbox("Drop Dye", &mut self.draw_dye);
                    ui.separator();

                    ui.text("Input");
                    ui.set_next_item_width(-100.0);
                    imgui::Slider::new("Draw Size", 0.0, 100.0)
                        .display_format("%.2f")
                        .build(&ui, &mut self.draw_size);
                    ui.set_next_item_width(-100.0);
                    imgui::Slider::new("Push Strength", 0.0, 10.0)
                        .display_format("%.2f")
                        .build(&ui, &mut self.push_power);
                    imgui::ColorPicker::new("Dye Color", &mut self.draw_color)
                        .build(&ui);
                    ui.separator();

                    ui.text("Parameters");
                    ui.set_next_item_width(-100.0);
                    imgui::Slider::new("Vorticity", 0.0, 5.0)
                        .display_format("%.2f")
                        .build(&ui, &mut self.vort);
                    ui.separator();
                    ui.text("Not Yet Implemented");
                    imgui::Slider::new("Viscosity", 0.0, 5.0)
                        .display_format("%.2f")
                        .build(&ui, &mut 0.0);
                    imgui::Slider::new("Quality", 5, 100)
                        .display_format("%d")
                        .build(&ui, &mut 50);
                    imgui::Slider::new("Timestep (FPS)", 30, 144)
                        .display_format("%d")
                        .build(&ui, &mut 100);
                    ui.text("- Pressure Poisson")
                });
            });
        }
        self.imgui_platform.prepare_render(&ui, window);
        let swapchain_texture_view = swapchain_texture
            .texture
            .create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
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
        self.imgui_renderer
            .render(ui.render(), &self.queue, &self.device, &mut render_pass)
            .unwrap();
        drop(render_pass);
        self.queue.submit([encoder.finish()]);
        swapchain_texture.present();
    }

    fn get_push_const(&self, stage: Stage) -> PushConst {
        let force_pos = (
            self.mouse_pos.0 as i32 * SIM_SIZE as i32 / self.surface_config.width as i32,
            self.mouse_pos.1 as i32 * SIM_SIZE as i32 / self.surface_config.height as i32,
        );
        let del_len = (self.mouse_del.0 * self.mouse_del.0 + self.mouse_del.1 * self.mouse_del.1).sqrt();
        let force_dir = (
            self.push_power * (del_len * del_len * self.mouse_del.0) as f32,
            self.push_power * (del_len * del_len * self.mouse_del.1) as f32,
        );
        let over_ui = self.imgui_context.io().want_capture_mouse;
        PushConst {
            draw_color: self.draw_color,
            dimension: (SIM_SIZE as u32, SIM_SIZE as u32),
            force_pos,
            force_dir,
            draw_size: self.draw_size,
            dt_s: self.timestep.as_secs_f32(),
            vort: self.vort,
            stage: stage as u32,
            draw_dye: (!over_ui && self.draw_dye && self.mouse_down) as u32,
            push_vel: (!over_ui && self.push_vel && self.mouse_down) as u32,
        }
    }

    fn clear_buffers(&mut self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.vel_tmp_buff, 0, None);
        encoder.clear_buffer(&self.dye_tmp_buff, 0, None);
        encoder.clear_buffer(&self.tmp_buff_0, 0, None);
        encoder.clear_buffer(&self.tmp_buff_1, 0, None);
    }

    fn update(&mut self) {
        let (dispatch_x, dispatch_y) =
            util::dispatch_size((SIM_SIZE, SIM_SIZE), (WG_SIZE, WG_SIZE));
        let mut encoder = self
            .device
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
            let timesteps = if self.accum_time > self.timestep {
                self.accum_time -= self.timestep;
                1
            } else {
                0
            };
            for _ in 0..timesteps {
                compute_pass.set_push_constants(0, to_raw(&[self.get_push_const(Stage::Input)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                compute_pass
                    .set_push_constants(0, to_raw(&[self.get_push_const(Stage::AdvectVel)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                compute_pass.set_push_constants(0, to_raw(&[self.get_push_const(Stage::SwapVel)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                compute_pass.set_push_constants(0, to_raw(&[self.get_push_const(Stage::VortConf)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                compute_pass.set_push_constants(0, to_raw(&[self.get_push_const(Stage::SwapVel)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                for _ in 0..QUAL {
                    compute_pass
                        .set_push_constants(0, to_raw(&[self.get_push_const(Stage::Project)]));
                    compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                    compute_pass
                        .set_push_constants(0, to_raw(&[self.get_push_const(Stage::SwapTmp)]));
                    compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                }
                compute_pass.set_push_constants(0, to_raw(&[self.get_push_const(Stage::RemDiv)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                compute_pass
                    .set_push_constants(0, to_raw(&[self.get_push_const(Stage::AdvectDye)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
                compute_pass.set_push_constants(0, to_raw(&[self.get_push_const(Stage::SwapDye)]));
                compute_pass.dispatch(dispatch_x, dispatch_y, 1);
            }
        }
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
        self.queue.submit([encoder.finish()]);
    }

    fn perf_update(&mut self) {
        self.elapsed_time = self.current_time.elapsed();
        self.current_time = Instant::now();
        self.accum_time += self.elapsed_time;
        self.imgui_context
            .io_mut()
            .update_delta_time(self.elapsed_time);
    }

    fn handle_window_event(&mut self, event: &WindowEvent) -> ControlFlow {
        match event {
            WindowEvent::CloseRequested => return ControlFlow::Exit,
            WindowEvent::Resized(size) => self.resize_surface(*size),
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_del = (position.x - self.mouse_pos.0, position.y - self.mouse_pos.1);
                self.mouse_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput { button, state, .. } => match button {
                winit::event::MouseButton::Left => self.mouse_down = *state == ElementState::Pressed,
                _ => {}
            },
            _ => {}
        };
        ControlFlow::Poll
    }

    fn handle_event(&mut self, event: Event<()>, control_flow: &mut ControlFlow, window: &Window) {
        self.imgui_platform
            .handle_event(self.imgui_context.io_mut(), &window, &event);
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => *control_flow = self.handle_window_event(event),
            Event::MainEventsCleared => {
                self.perf_update();
                self.update();
                self.render(&window);
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new(); // TODO: Move into App
    let window = WindowBuilder::new()
        .with_title("Fluid")
        .with_inner_size(PhysicalSize::new(WINDOW_SIZE, WINDOW_SIZE))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();
    let mut app = App::new(&window);
    event_loop.run(move |event, _, control_flow| {
        app.handle_event(event, control_flow, &window)
    });
}
