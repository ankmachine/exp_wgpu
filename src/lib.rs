use std::sync::Arc;

use winit::{
    application::ApplicationHandler, event::*, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::Window
};

pub struct State {
    window : Arc<Window>
}

impl State {
    pub fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        Ok(Self{
            window,
        })
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        TODO!(self);
    }
    pub fn render(&mut self) {
        TODO!(self);
    }
}

pub struct App {
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(event_loop: &EventLoop<State>) -> Self {
        let proxy = Some(event_loop.create_proxy());
        Self{
            state: None,
            proxy,
        }
    }
}