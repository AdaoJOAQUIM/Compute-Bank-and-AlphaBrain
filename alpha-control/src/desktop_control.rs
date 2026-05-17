// Axiom P — Desktop accessibility control
// Linux: AT-SPI via atspi-common crate (future)
// macOS: NSAppleScript (future)
// Windows: UIAutomation COM (future)

pub struct DesktopControl {
    pub platform: Platform,
    pub action_log: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Platform { Linux, MacOS, Windows, Unknown }

impl DesktopControl {
    pub fn new() -> Self {
        let platform = if cfg!(target_os = "linux") { Platform::Linux }
            else if cfg!(target_os = "macos") { Platform::MacOS }
            else if cfg!(target_os = "windows") { Platform::Windows }
            else { Platform::Unknown };
        Self { platform, action_log: Vec::new() }
    }

    pub fn open_app(&mut self, name: &str) -> Result<(), String> {
        self.action_log.push(format!("open:{}", name));
        Ok(()) // stub — real impl: AT-SPI launch
    }

    pub fn type_text(&mut self, text: &str) -> Result<(), String> {
        self.action_log.push(format!("type:{}", text));
        Ok(())
    }

    pub fn take_screenshot(&mut self) -> Result<Vec<u8>, String> {
        self.action_log.push("screenshot".to_string());
        Ok(vec![]) // stub
    }
}

impl Default for DesktopControl { fn default() -> Self { Self::new() } }
