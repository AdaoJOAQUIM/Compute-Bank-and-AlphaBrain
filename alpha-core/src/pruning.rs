//! Axiom M — Entropic pruning.
//! score(ξ) = α·freq + β·basin + γ·recency
//! Never prune core basins or active S-expressions.

use crate::sparse_w::SparseW;
use crate::sparse_vec::SparseVec;

const ALPHA: f64 = 0.4;
const BETA:  f64 = 0.4;
const GAMMA: f64 = 0.2;

pub struct PatternMeta {
    pub pattern:       SparseVec,
    pub access_count:  u64,
    pub basin_radius:  f64,
    pub last_tick:     u64,
    pub is_core:       bool,  // never prune
    pub is_sexpr:      bool,  // never prune if active evals pending
    pub pending_evals: u32,
}

impl PatternMeta {
    pub fn score(&self, now_tick: u64) -> f64 {
        let recency = 1.0 / (1.0 + (now_tick - self.last_tick) as f64);
        ALPHA * self.access_count as f64 + BETA * self.basin_radius + GAMMA * recency
    }
}

pub struct EntropicPruner {
    pub patterns: Vec<PatternMeta>,
    threshold:    f64,
}

impl EntropicPruner {
    pub fn new(threshold: f64) -> Self { Self { patterns: Vec::new(), threshold } }

    /// Prune patterns below threshold. Preserves core and active S-expressions.
    pub fn prune(&mut self, w: &mut SparseW, now_tick: u64) -> usize {
        let threshold = self.threshold;
        let mut pruned = 0;
        self.patterns.retain(|p| {
            let keep = p.is_core
                || (p.is_sexpr && p.pending_evals > 0)
                || p.score(now_tick) >= threshold;
            if !keep { pruned += 1; }
            keep
        });
        // Re-derive W from remaining patterns
        *w = SparseW::new();
        let scale = 1.0 / (self.patterns.len().max(1) as f32);
        for p in &self.patterns {
            w.hebbian_store(&p.pattern, scale);
        }
        pruned
    }
}
