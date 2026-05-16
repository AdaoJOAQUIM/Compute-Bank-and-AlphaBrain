// Stigmergic coordination field  Φ  — Axiom D.
//
// Continuous-time PDE:  ∂Φ/∂t = D·∇²Φ + source − decay·Φ
//
// Discretised (local Laplacian, Axiom B — no global state):
//   ∇²Φ(x) ≈ mean(Φ(neighbours)) − Φ(x)
//
// Two pheromone channels mirroring the research document:
//   Φ₁ — work-available signal (topology layer)
//   Φ₂ — memory-pressure signal (consolidation trigger)

#[derive(Clone, Debug)]
pub struct Pheromone {
    pub phi1: f64,      // work-available concentration
    pub phi2: f64,      // memory-pressure concentration
    pub diffusion: f64, // D coefficient
    pub decay1: f64,    // decay rate for Φ₁
    pub decay2: f64,    // decay rate for Φ₂
}

impl Default for Pheromone {
    fn default() -> Self {
        Self {
            phi1: 0.0,
            phi2: 0.0,
            diffusion: 0.1,
            decay1: 0.05,
            decay2: 0.02,
        }
    }
}

impl Pheromone {
    pub fn new() -> Self { Self::default() }

    /// Advance one time-step given neighbour values and local source terms.
    pub fn step(
        &mut self,
        neighbor_phi1: &[f64],
        neighbor_phi2: &[f64],
        source1: f64,
        source2: f64,
        dt: f64,
    ) {
        let lap1 = laplacian(self.phi1, neighbor_phi1);
        let lap2 = laplacian(self.phi2, neighbor_phi2);
        self.phi1 += dt * (self.diffusion * lap1 + source1 - self.decay1 * self.phi1);
        self.phi2 += dt * (self.diffusion * lap2 + source2 - self.decay2 * self.phi2);
        self.phi1 = self.phi1.max(0.0);
        self.phi2 = self.phi2.max(0.0);
    }
}

fn laplacian(local: f64, neighbors: &[f64]) -> f64 {
    if neighbors.is_empty() { return 0.0; }
    let mean: f64 = neighbors.iter().sum::<f64>() / neighbors.len() as f64;
    mean - local
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_stays_non_negative() {
        let mut p = Pheromone::new();
        // Large negative source should not drive phi below zero
        for _ in 0..100 {
            p.step(&[], &[], -1000.0, -1000.0, 0.1);
        }
        assert!(p.phi1 >= 0.0);
        assert!(p.phi2 >= 0.0);
    }

    #[test]
    fn phi_decays_to_zero_without_source() {
        // Verify exponential decay: after n steps, phi ≈ phi0·(1 − decay·dt)^n.
        // phi1: decay=0.05, phi2: decay=0.02 — both strictly decreasing.
        let mut p = Pheromone::new();
        p.phi1 = 1.0;
        p.phi2 = 1.0;
        for _ in 0..1000 {
            p.step(&[], &[], 0.0, 0.0, 0.1);
        }
        // phi1: (1−0.005)^1000 ≈ 0.0067  < 0.01 ✓
        assert!(p.phi1 < 0.01, "phi1 should decay to ~0, got {}", p.phi1);
        // phi2: (1−0.002)^1000 ≈ 0.135   — strictly below initial value of 1.0
        assert!(p.phi2 < 0.20, "phi2 should decay from 1.0, got {}", p.phi2);
        assert!(p.phi2 > 0.0,  "phi2 must stay non-negative, got {}", p.phi2);
    }
}
