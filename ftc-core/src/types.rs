// Physical constants and primitive types — Axiom A: every variable = measurable quantity.

pub const D: usize = 256;              // field dimension (global const per spec)
pub const ETA: f32  = 0.01;            // Hebbian learning rate η
pub const LAMBDA: f32 = 0.001;         // weight decay λ
pub const BETA: f64 = 4.0;            // inverse temperature for Modern Hopfield
pub const K_B: f64  = 1.380_649e-23;  // Boltzmann constant  J·K⁻¹
pub const T_ROOM: f64 = 293.15;        // room temperature    K
pub const ENERGY_PER_FLOP: f64 = 1e-15; // 1 fJ per FLOP (conservative hardware estimate)

/// Theoretical Landauer minimum: k_B · T · ln(2)  ≈ 2.8 × 10⁻²¹ J per bit erased.
pub const LANDAUER_MIN: f64 = K_B * T_ROOM * std::f64::consts::LN_2;

/// Physical energy in joules (simulated via FLOP accounting — Axiom A).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Energy(pub f64);

impl std::ops::Add for Energy {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Energy(self.0 + rhs.0) }
}

impl std::ops::AddAssign for Energy {
    fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
}

/// Opaque node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "N{}", self.0)
    }
}
