use wgpu::{Backends, Device, DeviceDescriptor, Features, Instance, Limits, PowerPreference, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration};
use winit::{dpi::PhysicalSize, event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window};

struct App {
    surface:    Surface,
    device:     Device,
    queue:      Queue,
    config:     SurfaceConfiguration,
    size:       PhysicalSize<u32>,
}

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
        surface.configure(&device, &config);
        App {
            surface, device, queue, config, size
        }
    }
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }
    pub fn render(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1, // Pick any color you want here
                            g: 0.9,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
        }
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
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
            Event::RedrawRequested(window_id) if window_id == window.id() => app.render(),
            _ => {}
        }
    });
}
