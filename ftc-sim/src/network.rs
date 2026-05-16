use std::collections::{HashMap, HashSet};
use ftc_core::{NodeId, D};
use crate::node::Node;

/// The distributed runtime: a graph of nodes connected by physical links.
/// No global coordinator — each node acts only on local information (Axiom B).
pub struct Network {
    pub nodes: HashMap<NodeId, Node>,
    pub tick: u64,
    dead: HashSet<NodeId>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            tick: 0,
            dead: HashSet::new(),
        }
    }

    pub fn add_node(&mut self, id: NodeId) {
        self.nodes.insert(id, Node::new(id));
    }

    /// Bidirectional edge (physical link).
    pub fn connect(&mut self, a: NodeId, b: NodeId) {
        if let Some(n) = self.nodes.get_mut(&a) { n.neighbors.push(b); }
        if let Some(n) = self.nodes.get_mut(&b) { n.neighbors.push(a); }
    }

    pub fn kill_node(&mut self, id: NodeId) {
        self.dead.insert(id);
    }

    pub fn revive_node(&mut self, id: NodeId) {
        self.dead.remove(&id);
    }

    pub fn is_alive(&self, id: NodeId) -> bool {
        !self.dead.contains(&id)
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        if self.is_alive(id) { self.nodes.get(&id) } else { None }
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        if !self.dead.contains(&id) { self.nodes.get_mut(&id) } else { None }
    }

    pub fn alive_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().filter(|&id| self.is_alive(id)).collect()
    }

    pub fn total_nodes(&self) -> usize { self.nodes.len() }
    pub fn alive_count(&self) -> usize { self.alive_ids().len() }
    pub fn dead_count(&self) -> usize  { self.dead.len() }

    /// Total energy consumed across all nodes (joules — Axiom A).
    pub fn total_energy(&self) -> f64 {
        self.nodes.values().map(|n| n.energy_consumed).sum()
    }

    /// Total patterns stored (with multiplicity across replicas).
    pub fn total_stored_patterns(&self) -> usize {
        self.nodes.values().map(|n| n.patterns_stored).sum()
    }

    /// Pheromone diffusion tick (Axiom D — local Laplacian, no global state).
    pub fn pheromone_tick(&mut self, dt: f64) {
        // Snapshot phi values before updating so all nodes see t not t+1
        let phi_snap: HashMap<NodeId, (f64, f64)> = self.nodes.iter()
            .map(|(&id, n)| (id, (n.pheromone.phi1, n.pheromone.phi2)))
            .collect();

        for (&id, node) in self.nodes.iter_mut() {
            if self.dead.contains(&id) { continue; }
            let (nbr_phi1, nbr_phi2): (Vec<f64>, Vec<f64>) = node.neighbors.iter()
                .filter_map(|nid| phi_snap.get(nid).copied())
                .unzip();
            node.pheromone.step(&nbr_phi1, &nbr_phi2, 0.0, 0.0, dt);
        }
        self.tick += 1;
    }

    // ── Topology builders ────────────────────────────────────────────────────

    /// Ring graph — diameter = N/2.
    pub fn ring(n: usize) -> Self {
        let mut net = Self::new();
        for i in 0..n { net.add_node(NodeId(i as u64)); }
        for i in 0..n {
            net.connect(NodeId(i as u64), NodeId(((i + 1) % n) as u64));
        }
        net
    }

    /// Erdős–Rényi random graph — expected degree = edge_prob × (n−1).
    pub fn random(n: usize, edge_prob: f64, rng: &mut impl rand::Rng) -> Self {
        let mut net = Self::new();
        for i in 0..n { net.add_node(NodeId(i as u64)); }
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.gen::<f64>() < edge_prob {
                    net.connect(NodeId(i as u64), NodeId(j as u64));
                }
            }
        }
        net
    }

    /// Full mesh — every node connected to every other.  O(N²) edges.
    pub fn full_mesh(n: usize) -> Self {
        Self::random(n, 1.0, &mut rand::thread_rng())
    }

    /// Two-group topology: dense within groups, sparse between — simulates
    /// cross-provider topology (Phase III / E5).
    pub fn two_groups(n_per_group: usize, intra_prob: f64, inter_prob: f64,
                      rng: &mut impl rand::Rng) -> Self {
        let n = 2 * n_per_group;
        let mut net = Self::new();
        for i in 0..n { net.add_node(NodeId(i as u64)); }
        for i in 0..n {
            for j in (i + 1)..n {
                let same_group = (i < n_per_group) == (j < n_per_group);
                let p = if same_group { intra_prob } else { inter_prob };
                if rng.gen::<f64>() < p {
                    net.connect(NodeId(i as u64), NodeId(j as u64));
                }
            }
        }
        net
    }

    // ── Workload helpers (topology tests E1–E6) ──────────────────────────────

    /// Process a computation represented by field `task` on node `id`.
    /// Adapts W via Axiom C and emits Φ₁ pheromone proportional to load.
    pub fn process_task(&mut self, id: NodeId, task: &[f32; D]) -> Option<f64> {
        let node = self.get_node_mut(id)?;
        node.adapt(task);
        node.pheromone.phi1 += 1.0; // work-available signal
        Some(node.energy_consumed)
    }
}

impl Default for Network {
    fn default() -> Self { Self::new() }
}
