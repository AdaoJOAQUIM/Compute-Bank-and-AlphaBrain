// Semantic Gradient Routing (SGR) — Axiom F.
//
// Retrieval algorithm:
//   1. Start at node x with query ξ_q.
//   2. Run one local Hopfield step (refine ξ_q toward nearest local attractor).
//   3. Compute local energy E_x = −½ ξᵀ W_x ξ / d.
//   4. Ask each (alive, unvisited) neighbour for their energy E_j.
//   5. If min(E_j) < E_x → migrate to argmin_j E_j, increment hop count.
//   6. Else → local minimum reached; run full Hopfield relax and return.
//
// No directory, no coordinator, no addressed messages — Axiom B guaranteed.
// Termination: visited-set prevents cycles; max_hops hard cap.

use std::collections::HashSet;
use ftc_core::{NodeId, D};
use crate::network::Network;

pub struct SgrResult {
    /// Recovered pattern after local Hopfield relaxation.
    pub recovered: Box<[f32; D]>,
    /// Number of inter-node hops taken.
    pub hops: usize,
    /// Sequence of nodes visited (for path analysis).
    pub path: Vec<NodeId>,
    /// True if routing reached a proper local minimum (not cut off by max_hops).
    pub converged: bool,
}

/// Retrieve the pattern closest to `query` via Semantic Gradient Routing.
///
/// `start` — the node to begin search from (caller picks any alive node).
/// `max_hops` — hard cap; E10 uses 50, large experiments use 3·log₂(N).
pub fn sgr_retrieve(
    query: &[f32; D],
    start: NodeId,
    network: &Network,
    max_hops: usize,
) -> SgrResult {
    let mut cur_node = start;
    let mut state: Box<[f32; D]> = {
        let mut s = Box::new([0.0f32; D]);
        s.copy_from_slice(query);
        s
    };
    let mut path = vec![start];
    let mut visited: HashSet<NodeId> = HashSet::new();
    visited.insert(start);

    // Route using the ORIGINAL query throughout — the Hopfield step must only
    // happen at the final destination node.  Modifying state mid-route biases
    // the energy landscape toward the starting node's attractors and prevents
    // gradient following (this was the key design error to avoid).
    'routing: for _ in 0..max_hops {
        let node = match network.get_node(cur_node) {
            Some(n) => n,
            None => break 'routing,
        };

        // Energy of the ORIGINAL query at this node.
        let local_e = node.energy_for(&state);

        // Find best unvisited alive neighbour, also evaluated on original query.
        let best = node.neighbors.iter()
            .filter(|&&nid| !visited.contains(&nid) && network.is_alive(nid))
            .filter_map(|&nid| {
                network.get_node(nid)
                    .map(|n| (nid, n.energy_for(&state)))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((nid, e)) if e < local_e => {
                visited.insert(nid);
                cur_node = nid;
                path.push(nid);
            }
            _ => break 'routing, // genuine local minimum in the W-field
        }
    }

    let converged = path.len() <= max_hops;

    // Final Hopfield relaxation happens only at the destination — this is where
    // the query is "interpreted" by the local attractor landscape.
    let recovered = match network.get_node(cur_node) {
        Some(n) => n.w.hopfield_relax(&state, 20),
        None    => state,
    };

    SgrResult {
        recovered,
        hops: path.len().saturating_sub(1),
        path,
        converged,
    }
}

/// Batch retrieve: run SGR for each (query, start_node) pair and
/// return (mean hops, success fraction) relative to `originals`.
pub fn batch_retrieve(
    queries: &[(&[f32; D], NodeId)],
    originals: &[&[f32; D]],
    network: &Network,
    max_hops: usize,
    success_threshold: f64,
) -> (f64, f64) {
    assert_eq!(queries.len(), originals.len());
    let mut total_hops = 0usize;
    let mut successes = 0usize;

    for ((q, start), orig) in queries.iter().zip(originals.iter()) {
        let res = sgr_retrieve(q, *start, network, max_hops);
        total_hops += res.hops;
        let sim = ftc_core::bit_similarity(&res.recovered, orig);
        if sim >= success_threshold { successes += 1; }
    }

    let n = queries.len().max(1);
    (total_hops as f64 / n as f64, successes as f64 / n as f64)
}
