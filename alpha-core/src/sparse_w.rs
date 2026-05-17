use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use crate::sparse_vec::SparseVec;
use crate::types::{ETA, LAMBDA, K};

/// Sparse symmetric W matrix, zero diagonal.
/// Stores only (i, j) with i < j.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseW {
    entries: BTreeMap<(u16, u16), f32>,
    pub n_patterns: usize,
}

impl SparseW {
    pub fn new() -> Self { Self::default() }

    pub fn get(&self, i: u16, j: u16) -> f32 {
        if i == j { return 0.0; }
        let key = (i.min(j), i.max(j));
        *self.entries.get(&key).unwrap_or(&0.0)
    }

    fn set_raw(&mut self, i: u16, j: u16, v: f32) {
        if i == j { return; }
        let key = (i.min(j), i.max(j));
        if v.abs() < 1e-9 { self.entries.remove(&key); }
        else              { self.entries.insert(key, v); }
    }

    /// Hebbian store: ΔW(a,b) += scale × ξ(a) × ξ(b)  [Axiom G storage rule]
    pub fn hebbian_store(&mut self, xi: &SparseVec, scale: f32) {
        for (ii, &i) in xi.indices.iter().enumerate() {
            for (jj, &j) in xi.indices.iter().enumerate() {
                if i >= j { continue; }
                let delta = scale * xi.values[ii] * xi.values[jj];
                let key = (i, j);
                *self.entries.entry(key).or_insert(0.0) += delta;
            }
        }
        self.n_patterns += 1;
    }

    /// Axiom C: ΔW = η·(Ψ⊗Ψ) − λ·W  (sparse, only active pairs)
    pub fn plasticity_step(&mut self, psi: &SparseVec) {
        // Decay all entries by λ
        for v in self.entries.values_mut() { *v *= 1.0 - LAMBDA; }
        // Reinforce active pairs by η
        for (ii, &i) in psi.indices.iter().enumerate() {
            for (jj, &j) in psi.indices.iter().enumerate() {
                if i >= j { continue; }
                let key = (i, j);
                *self.entries.entry(key).or_insert(0.0) += ETA * psi.values[ii] * psi.values[jj];
            }
        }
    }

    /// Hopfield energy: E(ψ) = −½ Σ_{active i<j} W[i,j]·ψ[i]·ψ[j] / K
    pub fn energy(&self, psi: &SparseVec) -> f64 {
        let mut e = 0.0f64;
        for (ii, &i) in psi.indices.iter().enumerate() {
            for (jj, &j) in psi.indices.iter().enumerate() {
                if i >= j { continue; }
                e += self.get(i, j) as f64 * psi.values[ii] as f64 * psi.values[jj] as f64;
            }
        }
        -0.5 * e / K as f64
    }

    /// One synchronous Hopfield step: same active indices, updated signs.
    pub fn hopfield_step(&self, psi: &SparseVec) -> SparseVec {
        let values = psi.indices.iter().enumerate().map(|(ii, &i)| {
            let sum: f32 = psi.indices.iter().enumerate()
                .filter(|(_, &j)| j != i)
                .map(|(jj, &j)| self.get(i, j) * psi.values[jj])
                .sum();
            if sum >= 0.0 { 1.0f32 } else { -1.0f32 }
        }).collect();
        SparseVec { indices: psi.indices.clone(), values }
    }

    pub fn hopfield_relax(&self, psi: &SparseVec, max_steps: usize) -> SparseVec {
        let mut cur = psi.clone();
        for _ in 0..max_steps {
            let nxt = self.hopfield_step(&cur);
            if nxt == cur { return nxt; }
            cur = nxt;
        }
        cur
    }

    /// W-CRDT merge: confidence-weighted average [Axiom F distributed merge]
    pub fn merge(&self, other: &SparseW, self_weight: f64, other_weight: f64) -> SparseW {
        let total = (self_weight + other_weight) as f32;
        let wa = self_weight as f32 / total;
        let wb = other_weight as f32 / total;
        let mut merged = SparseW::new();
        for (&k, &v) in &self.entries  { *merged.entries.entry(k).or_insert(0.0) += wa * v; }
        for (&k, &v) in &other.entries { *merged.entries.entry(k).or_insert(0.0) += wb * v; }
        merged.entries.retain(|_, v| v.abs() >= 1e-9);
        merged.n_patterns = (self.n_patterns + other.n_patterns + 1) / 2;
        merged
    }

    /// Frobenius cosine similarity
    pub fn coupling(&self, other: &SparseW) -> f64 {
        let dot: f64 = self.entries.iter()
            .filter_map(|(k, &a)| other.entries.get(k).map(|&b| a as f64 * b as f64))
            .sum();
        let na: f64 = self.entries.values().map(|&v| (v*v) as f64).sum::<f64>().sqrt();
        let nb: f64 = other.entries.values().map(|&v| (v*v) as f64).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 { 0.0 } else { (dot / (na * nb)).clamp(-1.0, 1.0) }
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }

    /// Access entries for testing / inspection
    pub fn entries(&self) -> &BTreeMap<(u16, u16), f32> { &self.entries }
}

// Suppress unused warning for set_raw while keeping it for future use
#[allow(dead_code)]
fn _use_set_raw(w: &mut SparseW) { w.set_raw(0, 1, 0.0); }
