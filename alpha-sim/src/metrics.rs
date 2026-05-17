//! CPU profiling and adiabatic invariant metrics.
//! Simulates ARM Cortex-A72 energy model when perf_event is unavailable.

use alpha_core::{NodeId, LANDAUER_MIN};
use crate::network::AlphaNetwork;

/// Simulated ARM CPU profile for one operation.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ArmCpuProfile {
    pub ops:          u64,
    pub time_us:      f64,   // microseconds
    pub energy_nj:    f64,   // nanojoules (estimated)
    pub cpu_util:     f64,   // [0, 1]
}

impl ArmCpuProfile {
    /// Model: ARM Cortex-A72, ~1.5 GHz effective, ~3 W TDP for 4 cores → 0.75 W per core
    pub fn from_ops(ops: u64) -> Self {
        let ghz       = 1.5;
        let time_us   = ops as f64 / (ghz * 1e3);  // μs
        let power_w   = 0.75;  // one core at full load
        let energy_nj = power_w * time_us * 1e-6 * 1e9;  // W × s × 1e9 = nJ
        let cpu_util  = (ops as f64 / (ghz * 1e6 * 100.0)).clamp(0.0, 1.0); // vs 100 ms budget
        Self { ops, time_us, energy_nj, cpu_util }
    }

    pub fn energy_joules(&self) -> f64 { self.energy_nj * 1e-9 }
}

/// Φ₁ source from CPU ops (Axiom K proxy).
pub fn phi1_source(ops: u64, cpu_budget: u64) -> f64 {
    (ops as f64 / cpu_budget as f64).clamp(0.0, 1.0)
}

/// Estimate energy per stored bit in the network.
pub fn energy_per_bit(net: &AlphaNetwork) -> f64 {
    let total_energy: f64 = net.nodes.values()
        .map(|n| ArmCpuProfile::from_ops(n.ops_this_tick).energy_joules())
        .sum();
    let total_bits: f64 = net.nodes.values()
        .map(|n| n.patterns_stored as f64 * alpha_core::K as f64)
        .sum::<f64>()
        .max(1.0);
    total_energy / total_bits
}

/// Landauer ratio: energy_per_bit / LANDAUER_MIN
pub fn landauer_ratio(net: &AlphaNetwork) -> f64 {
    let epb = energy_per_bit(net);
    if epb <= 0.0 { 1.0 } else { epb / LANDAUER_MIN }
}
