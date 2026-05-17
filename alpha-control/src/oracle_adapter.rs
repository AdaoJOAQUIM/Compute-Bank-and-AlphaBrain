// Optional cloud adapter — Oracle Always-Free ARM64
// One OAuth. W_disk backup when laptop offline.
// Real impl: OCI SDK LaunchInstanceRequest
// Stub: serializes W to JSON for mock sync

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Default)]
pub struct OracleConfig {
    pub tenancy_ocid: String,
    pub region: String,
    pub compartment_id: String,
}

pub struct OracleAdapter {
    pub config: OracleConfig,
    pub connected: bool,
}

impl OracleAdapter {
    pub fn new(config: OracleConfig) -> Self { Self { config, connected: false } }

    pub fn connect(&mut self) -> Result<(), String> {
        // Real impl: OCI auth + verify ARM64 instance
        self.connected = true;
        Ok(())
    }

    // Sync W snapshot to Oracle ARM (stub)
    pub fn push_w_snapshot(&self, w_json: &str) -> Result<(), String> {
        if !self.connected { return Err("not connected".to_string()); }
        let _ = w_json; // stub
        Ok(())
    }

    // Pull W snapshot from Oracle ARM (stub)
    pub fn pull_w_snapshot(&self) -> Result<String, String> {
        if !self.connected { return Err("not connected".to_string()); }
        Ok("{}".to_string()) // stub empty W
    }
}
