use ftc_core::{NodeId, WeightMatrix, Pheromone, FieldVec, D, ENERGY_PER_FLOP};

/// A single compute node in the Field Theory runtime.
///
/// `w` is the unified object: it simultaneously encodes topology coupling,
/// associative memory, and the SGR routing gradient — Unification Theorem §1.
pub struct Node {
    pub id: NodeId,
    pub w: WeightMatrix,
    pub pheromone: Pheromone,
    pub neighbors: Vec<NodeId>,
    pub energy_consumed: f64,   // joules (Axiom A FLOP accounting)
    pub patterns_stored: usize, // patterns whose Hebbian trace lives here
    pub access_count: u64,      // total retrieval queries served
}

impl Node {
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            w: WeightMatrix::zeros(),
            pheromone: Pheromone::new(),
            neighbors: Vec::new(),
            energy_consumed: 0.0,
            patterns_stored: 0,
            access_count: 0,
        }
    }

    /// Store one pattern via Hebbian rule (Axiom G replication):
    ///   ΔW += (1 / k·d) · ξ⊗ξ
    pub fn store_pattern(&mut self, xi: &[f32; D], k: usize) {
        let scale = 1.0 / (k as f32 * D as f32);
        let cost = self.w.hebbian_store(xi, scale);
        self.w.enforce_symmetry();
        self.energy_consumed += cost;
        self.patterns_stored += 1;
    }

    /// Classical Hopfield energy for query `psi` — used by SGR gradient.
    ///   E = −½ ΨᵀWΨ / d
    pub fn energy_for(&self, psi: &[f32; D]) -> f64 {
        self.w.energy(psi)
    }

    /// One Hopfield relaxation step: Ψ_new = sign(W·Ψ).
    pub fn hopfield_step(&self, psi: &[f32; D]) -> FieldVec {
        self.energy_consumed_charge(); // side-effect-free; cost recorded separately
        self.w.hopfield_step(psi)
    }

    /// Relax to local attractor (up to 20 steps).
    pub fn hopfield_relax(&mut self, psi: &[f32; D]) -> FieldVec {
        let result = self.w.hopfield_relax(psi, 20);
        self.energy_consumed += (D * D) as f64 * ENERGY_PER_FLOP * 20.0;
        self.access_count += 1;
        result
    }

    /// Return energy cost of one D×D matrix-vector product without mutating state.
    fn energy_consumed_charge(&self) -> f64 {
        (D * D) as f64 * ENERGY_PER_FLOP
    }

    /// Hebbian plasticity triggered by computation (Axiom C).
    /// When a query `psi` passes through this node, W adapts.
    pub fn adapt(&mut self, psi: &[f32; D]) {
        let cost = self.w.hebbian_update(psi);
        self.w.enforce_symmetry();
        self.energy_consumed += cost;
    }

    /// Topology coupling to another node: cosine similarity of their W matrices.
    pub fn coupling_to(&self, other: &Node) -> f64 {
        self.w.coupling(&other.w)
    }
}
