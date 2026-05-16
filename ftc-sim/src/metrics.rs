// Measurement functions — all tied to physical quantities (Axiom A).

use ftc_core::{NodeId, LANDAUER_MIN};
use crate::network::Network;

// ── Per-network measurements ─────────────────────────────────────────────────

/// Total energy consumed across all alive nodes (joules).
pub fn network_energy_joules(net: &Network) -> f64 {
    net.nodes.values().map(|n| n.energy_consumed).sum()
}

/// Total bits stored (each pattern = D bits; each replica counted separately).
pub fn bits_stored(net: &Network) -> f64 {
    net.nodes.values().map(|n| n.patterns_stored as f64 * ftc_core::D as f64).sum()
}

/// Energy per bit stored (J / bit).
pub fn energy_per_bit(net: &Network) -> f64 {
    let bits = bits_stored(net);
    if bits < 1.0 { return f64::INFINITY; }
    network_energy_joules(net) / bits
}

/// Ratio of actual energy/bit to the theoretical Landauer minimum.
/// A ratio > 1 means "above minimum" (required by thermodynamics).
/// A ratio approaching 1 over time is the E10 emergence signal.
pub fn landauer_ratio(net: &Network) -> f64 {
    energy_per_bit(net) / LANDAUER_MIN
}

// ── Topology metrics ─────────────────────────────────────────────────────────

/// Shannon entropy of the query-access distribution over nodes.
/// Lower entropy = more specialised topology (E3 emergence test).
pub fn access_entropy(net: &Network) -> f64 {
    let counts: Vec<f64> = net.nodes.values()
        .filter(|n| net.is_alive(n.id))
        .map(|n| n.access_count as f64)
        .collect();
    let total: f64 = counts.iter().sum();
    if total < 1.0 { return 0.0; }
    -counts.iter()
        .map(|&c| { let p = c / total; if p > 0.0 { p * p.log2() } else { 0.0 } })
        .sum::<f64>()
}

/// Maximum possible entropy for a uniform distribution over k alive nodes.
pub fn max_entropy(alive_count: usize) -> f64 {
    if alive_count <= 1 { return 0.0; }
    (alive_count as f64).log2()
}

// ── Retrieval quality ─────────────────────────────────────────────────────────

pub struct RetrievalStats {
    pub mean_fidelity: f64,
    pub min_fidelity: f64,
    pub mean_hops: f64,
    pub success_rate: f64,   // fraction achieving >= 0.9 fidelity
}

impl RetrievalStats {
    pub fn compute(results: &[(f64, usize)]) -> Self {
        if results.is_empty() {
            return Self { mean_fidelity: 0.0, min_fidelity: 0.0, mean_hops: 0.0, success_rate: 0.0 };
        }
        let n = results.len() as f64;
        let mean_fidelity = results.iter().map(|r| r.0).sum::<f64>() / n;
        let min_fidelity  = results.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
        let mean_hops     = results.iter().map(|r| r.1 as f64).sum::<f64>() / n;
        let success_rate  = results.iter().filter(|r| r.0 >= 0.9).count() as f64 / n;
        Self { mean_fidelity, min_fidelity, mean_hops, success_rate }
    }
}

// ── Topology-beats-oracle (E1) ────────────────────────────────────────────────

/// Efficiency = patterns retrieved successfully / energy_consumed (normalised).
/// For the oracle comparison we use a fixed routing baseline: always start
/// from node 0 and take the shortest-path route (BFS), ignoring W gradients.
pub fn oracle_retrieval_hops(start: NodeId, network: &Network, max_hops: usize) -> usize {
    // Simplified oracle: BFS to the node with the most stored patterns.
    use std::collections::{VecDeque, HashSet};
    let target = network.nodes.values()
        .filter(|n| network.is_alive(n.id))
        .max_by_key(|n| n.patterns_stored)
        .map(|n| n.id);

    let target = match target {
        Some(t) => t,
        None => return max_hops,
    };
    if target == start { return 0; }

    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    queue.push_back((start, 0usize));
    visited.insert(start);

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops { break; }
        if let Some(n) = network.get_node(node) {
            for &nbr in &n.neighbors {
                if !visited.contains(&nbr) && network.is_alive(nbr) {
                    if nbr == target { return depth + 1; }
                    visited.insert(nbr);
                    queue.push_back((nbr, depth + 1));
                }
            }
        }
    }
    max_hops
}
