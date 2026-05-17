//! Axiom D/K — Pheromone fields Φ₁..Φ₅.
//! Φ₁ = CPU pressure (adiabatic invariant signal)
//! Φ₂ = memory pressure (STM→LTM trigger when Φ₂ < 0.2)
//! Φ₃ = error rate
//! Φ₄ = novelty
//! Φ₅ = task progress

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Pheromone {
    pub phi: [f64; 5],  // Φ₁..Φ₅
    pub diffusion: f64,
    pub decay: [f64; 5],
}

impl Pheromone {
    pub fn new() -> Self {
        Self {
            phi: [0.0; 5],
            diffusion: 0.1,
            decay: [0.05, 0.02, 0.10, 0.08, 0.03],
        }
    }

    /// One tick: Laplacian diffusion from neighbours + source injection + decay.
    pub fn step(&mut self, neighbor_phis: &[[f64; 5]], sources: [f64; 5], dt: f64) {
        let n_nbr = neighbor_phis.len() as f64;
        for ch in 0..5 {
            let mean_nbr = if n_nbr > 0.0 {
                neighbor_phis.iter().map(|p| p[ch]).sum::<f64>() / n_nbr
            } else { 0.0 };
            let laplacian = mean_nbr - self.phi[ch];
            self.phi[ch] += dt * (self.diffusion * laplacian + sources[ch] - self.decay[ch] * self.phi[ch]);
            self.phi[ch] = self.phi[ch].clamp(0.0, 1.0);
        }
    }

    pub fn phi1(&self) -> f64 { self.phi[0] }  // CPU pressure
    pub fn phi2(&self) -> f64 { self.phi[1] }  // memory pressure
    pub fn phi5(&self) -> f64 { self.phi[4] }  // task progress

    /// Adiabatic invariant check: CPU utilisation < θ
    pub fn adiabatic_ok(&self, theta: f64) -> bool { self.phi[0] < theta }

    /// STM→LTM consolidation trigger
    pub fn consolidate_now(&self) -> bool { self.phi[1] < 0.20 }
}

impl Default for Pheromone { fn default() -> Self { Self::new() } }
