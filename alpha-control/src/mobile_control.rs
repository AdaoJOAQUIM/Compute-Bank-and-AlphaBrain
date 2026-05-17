// Axiom P — Mobile device control
// Android: ADB shell commands (future real impl)
// iOS: XCTest companion over USB (future)

pub struct MobileControl {
    pub device_id: Option<String>,
    pub action_log: Vec<String>,
}

impl MobileControl {
    pub fn new(device_id: Option<String>) -> Self {
        Self { device_id, action_log: Vec::new() }
    }

    pub fn tap(&mut self, x: u32, y: u32) -> Result<(), String> {
        self.action_log.push(format!("tap:{},{}", x, y));
        Ok(())
    }

    pub fn swipe(&mut self, x1: u32, y1: u32, x2: u32, y2: u32) -> Result<(), String> {
        self.action_log.push(format!("swipe:{},{}->{},{}", x1, y1, x2, y2));
        Ok(())
    }

    pub fn screenshot(&mut self) -> Result<Vec<u8>, String> {
        self.action_log.push("screenshot".to_string());
        Ok(vec![])
    }
}
