use alpha_core::{NodeId, SparseW, SparseVec, Pheromone, K};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaNode {
    pub id:              NodeId,
    pub w:               SparseW,
    pub pheromone:       Pheromone,
    pub neighbors:       Vec<NodeId>,
    pub access_count:    u64,
    pub ops_this_tick:   u64,
    pub patterns_stored: usize,
    pub device_class:    DeviceClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceClass { Laptop, Phone, Cloud }

impl AlphaNode {
    pub fn new(id: NodeId, class: DeviceClass) -> Self {
        Self {
            id, w: SparseW::new(), pheromone: Pheromone::new(),
            neighbors: Vec::new(), access_count: 0, ops_this_tick: 0,
            patterns_stored: 0, device_class: class,
        }
    }

    /// Store pattern and track ops cost.
    pub fn store_pattern(&mut self, xi: &SparseVec, k: usize) {
        let scale = 1.0 / (k as f32 * K as f32);
        self.w.hebbian_store(xi, scale);
        self.patterns_stored += 1;
        self.ops_this_tick += (K * K) as u64; // O(k²)
    }

    /// Hopfield energy for a query at this node.
    pub fn energy_for(&self, psi: &SparseVec) -> f64 { self.w.energy(psi) }

    /// Axiom C plasticity adapt.
    pub fn adapt(&mut self, psi: &SparseVec) {
        self.w.plasticity_step(psi);
        self.ops_this_tick += (K * K) as u64;
    }

    /// Φ₁ source term: ops_this_tick / cpu_budget
    pub fn phi1_source(&self, cpu_budget: u64) -> f64 {
        (self.ops_this_tick as f64 / cpu_budget as f64).clamp(0.0, 1.0)
    }

    pub fn reset_tick_counters(&mut self) { self.ops_this_tick = 0; }
}
