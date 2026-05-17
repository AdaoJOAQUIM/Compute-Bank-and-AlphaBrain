use rand::seq::SliceRandom;
use alpha_core::{NodeId, SparseVec};
use crate::network::AlphaNetwork;
use crate::sgr::sgr_retrieve;

pub fn replication_factor(n: usize) -> usize {
    if n <= 1 { 1 } else { (n as f64).log2().ceil() as usize }
}

pub fn store_distributed(pattern: &SparseVec, k: usize, net: &mut AlphaNetwork, rng: &mut impl rand::Rng) -> Vec<NodeId> {
    let mut alive = net.alive_ids();
    alive.shuffle(rng);
    let chosen: Vec<NodeId> = alive.into_iter().take(k).collect();
    for &nid in &chosen {
        if let Some(node) = net.get_node_mut(nid) { node.store_pattern(pattern, k); }
    }
    chosen
}

pub fn retrieval_fidelity(original: &SparseVec, recovered: &SparseVec) -> f64 {
    original.bit_similarity(recovered)
}

pub fn measure_basin_radius(
    pattern: &SparseVec,
    start: NodeId,
    net: &AlphaNetwork,
    rng: &mut impl rand::Rng,
    trials: usize,
    threshold: f64,
) -> f64 {
    let noise_levels = [0.05f64, 0.10, 0.15, 0.20, 0.25, 0.30];
    let alive_count = net.alive_count();
    let max_hops = if alive_count <= 1 { 4 }
        else { ((alive_count as f64).log2().ceil() as usize) * 4 + 4 };
    let mut max_passing = 0.0f64;
    for &noise in &noise_levels {
        let successes = (0..trials).filter(|_| {
            let noisy = pattern.noisy(noise, rng);
            let res = sgr_retrieve(&noisy, start, net, max_hops);
            retrieval_fidelity(pattern, &res.recovered) >= threshold
        }).count();
        if successes as f64 / trials as f64 >= 0.7 { max_passing = noise; } else { break; }
    }
    max_passing
}
