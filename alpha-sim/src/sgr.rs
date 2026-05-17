//! Axiom F — Semantic Gradient Routing on sparse W.
//! Route using energy of ORIGINAL QUERY throughout.
//! Hopfield relax only at the destination.

use alpha_core::{NodeId, SparseVec};
use crate::network::AlphaNetwork;

pub struct SgrResult {
    pub recovered:  SparseVec,
    pub hops:       usize,
    pub path:       Vec<NodeId>,
    pub converged:  bool,
}

pub fn sgr_retrieve(query: &SparseVec, start: NodeId, net: &AlphaNetwork, max_hops: usize) -> SgrResult {
    let mut path = vec![start];
    let mut current = start;

    for _ in 0..max_hops {
        let node = match net.get_node(current) { Some(n) => n, None => break };
        let cur_energy = node.energy_for(query);

        // Check all alive neighbours
        let best_nbr = node.neighbors.iter()
            .filter(|&&nid| net.is_alive(nid))
            .filter_map(|&nid| net.get_node(nid).map(|n| (nid, n.energy_for(query))))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        match best_nbr {
            Some((nid, e)) if e < cur_energy => { current = nid; path.push(nid); }
            _ => break,  // local minimum — we're at the best reachable node
        }
    }

    // Hopfield relax at destination
    let recovered = if let Some(dest) = net.get_node(current) {
        dest.w.hopfield_relax(query, 20)
    } else {
        query.clone()
    };

    // converged = same as original (noisy with 0.0 = identity)
    let mut rng = rand::thread_rng();
    let identity = query.noisy(0.0, &mut rng);
    let converged = recovered == identity;

    SgrResult { recovered, hops: path.len().saturating_sub(1), path, converged }
}

/// Oracle retrieval: brute-force find node with minimum energy (O(N) scan).
pub fn oracle_retrieve(query: &SparseVec, net: &AlphaNetwork) -> SgrResult {
    let alive = net.alive_ids();
    if alive.is_empty() {
        return SgrResult { recovered: query.clone(), hops: 0, path: vec![], converged: false };
    }
    let best = alive.iter()
        .filter_map(|&id| net.get_node(id).map(|n| (id, n.energy_for(query))))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _)| id)
        .unwrap_or(alive[0]);

    let recovered = net.get_node(best).map(|n| n.w.hopfield_relax(query, 20)).unwrap_or_else(|| query.clone());
    let mut rng = rand::thread_rng();
    let identity = query.noisy(0.0, &mut rng);
    let converged = recovered == identity;

    SgrResult { recovered, hops: alive.len(), path: vec![best], converged }
}
