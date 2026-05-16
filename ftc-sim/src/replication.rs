// Holographic replication — Axiom G.
//
// k = ⌈log₂(N)⌉ replicas chosen uniformly at random.
// Storage rule: ΔW += (1 / k·d) · ξ⊗ξ  (Hebbian spreading).
// Retrieval fidelity degrades gracefully with replica loss:
//   ‖ξ_rec − ξ_orig‖ ≤ C·(1 − |S|/k)^½
// (Holographic Reconstruction Theorem §3.)

use rand::seq::SliceRandom;
use ftc_core::{NodeId, D};
use crate::network::Network;

/// Recommended replication factor for N alive nodes.
/// k = ⌈log₂(N)⌉, minimum 1.
pub fn replication_factor(n_nodes: usize) -> usize {
    if n_nodes <= 1 { return 1; }
    let k = (n_nodes as f64).log2().ceil() as usize;
    k.max(1)
}

/// Store `pattern` on `k` randomly chosen alive nodes using the Hebbian rule.
/// Returns the list of nodes that actually stored the pattern.
pub fn store_distributed(
    pattern: &[f32; D],
    k: usize,
    network: &mut Network,
    rng: &mut impl rand::Rng,
) -> Vec<NodeId> {
    let mut alive = network.alive_ids();
    alive.shuffle(rng);
    let chosen: Vec<NodeId> = alive.into_iter().take(k).collect();

    for &nid in &chosen {
        if let Some(node) = network.get_node_mut(nid) {
            node.store_pattern(pattern, k);
        }
    }
    chosen
}

/// How similar `recovered` is to `original` (fraction of bits matching).
pub fn retrieval_fidelity(original: &[f32; D], recovered: &[f32; D]) -> f64 {
    ftc_core::bit_similarity(original, recovered)
}

/// Attempt to measure basin radius around `pattern` at `node_id`.
///
/// Binary-search the maximum noise level at which SGR still recovers the
/// pattern with ≥ `success_threshold` accuracy, over `trials` repeated
/// noisy queries.  Returns the noise level (0..1).
pub fn measure_basin_radius(
    pattern: &[f32; D],
    node_id: NodeId,
    network: &Network,
    rng: &mut impl rand::Rng,
    trials: usize,
    success_threshold: f64,
) -> f64 {
    use ftc_core::noisy_field;
    use crate::sgr::sgr_retrieve;

    let noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40];
    let n = network.alive_count();
    let max_hops = (n as f64).log2().ceil() as usize * 4 + 4;

    let mut max_passing = 0.0f64;
    for &noise in &noise_levels {
        let mut successes = 0usize;
        for _ in 0..trials {
            let noisy = noisy_field(pattern, noise, rng);
            let res = sgr_retrieve(&noisy, node_id, network, max_hops);
            if retrieval_fidelity(pattern, &res.recovered) >= success_threshold {
                successes += 1;
            }
        }
        let rate = successes as f64 / trials as f64;
        if rate >= 0.7 {
            max_passing = noise;
        } else {
            break; // once success drops below 70 % the basin edge is here
        }
    }
    max_passing
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use ftc_core::{bipolar_field, noisy_field, bit_similarity};
    use crate::{network::Network, sgr::sgr_retrieve};

    #[test]
    fn replication_factor_grows_logarithmically() {
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let k = replication_factor(n);
            let upper = (n as f64).log2().ceil() as usize + 1;
            assert!(k <= upper, "k={k} exceeds log bound {upper} for n={n}");
        }
    }

    #[test]
    fn distributed_retrieval_basic() {
        let mut rng = StdRng::seed_from_u64(99);
        let n = 10;
        let mut net = Network::full_mesh(n);

        let pattern = bipolar_field(&mut rng);
        let k = replication_factor(n);
        store_distributed(&pattern, k, &mut net, &mut rng);

        // Retrieve from a random alive node with 10 % noise
        let start = NodeId(0);
        let noisy = noisy_field(&pattern, 0.10, &mut rng);
        let res = sgr_retrieve(&noisy, start, &net, 30);
        let sim = bit_similarity(&pattern, &res.recovered);
        assert!(sim >= 0.75, "basic retrieval fidelity too low: {sim:.3}");
    }
}
