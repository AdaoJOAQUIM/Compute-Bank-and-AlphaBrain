use serde::{Deserialize, Serialize};
use rand::Rng;
use crate::types::{D, K};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVec {
    pub indices: Vec<u16>,  // sorted, len == K
    pub values:  Vec<f32>,  // ±1.0 bipolar
}

impl SparseVec {
    pub fn new(indices: Vec<u16>, values: Vec<f32>) -> Self {
        debug_assert!(indices.windows(2).all(|w| w[0] < w[1]));
        debug_assert_eq!(indices.len(), values.len());
        Self { indices, values }
    }

    /// Random bipolar sparse vector with exactly K active bits.
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut idx: Vec<u16> = rand::seq::index::sample(rng, D, K)
            .iter().map(|i| i as u16).collect();
        idx.sort_unstable();
        let vals: Vec<f32> = idx.iter().map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
        Self { indices: idx, values: vals }
    }

    /// Flip each active bit independently with probability `noise`.
    pub fn noisy(&self, noise: f64, rng: &mut impl Rng) -> SparseVec {
        let values = self.values.iter().map(|&v| {
            if rng.gen::<f64>() < noise { -v } else { v }
        }).collect();
        SparseVec { indices: self.indices.clone(), values }
    }

    /// Fraction of shared active indices with matching sign (Jaccard variant).
    pub fn bit_similarity(&self, other: &SparseVec) -> f64 {
        let (mut matches, mut intersection) = (0usize, 0usize);
        let (mut i, mut j) = (0, 0);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    if (self.values[i] > 0.0) == (other.values[j] > 0.0) { matches += 1; }
                    i += 1; j += 1;
                }
                std::cmp::Ordering::Less    => { i += 1; }
                std::cmp::Ordering::Greater => { j += 1; }
            }
        }
        // Jaccard: union = |A| + |B| - |A ∩ B|
        let total_union = self.indices.len() + other.indices.len() - intersection;
        if total_union == 0 { return 0.0; }
        matches as f64 / total_union as f64
    }

    /// Fraction of overlapping active indices (set intersection / K).
    pub fn active_overlap(&self, other: &SparseVec) -> f64 {
        let (mut common, mut i, mut j) = (0usize, 0, 0);
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal   => { common += 1; i += 1; j += 1; }
                std::cmp::Ordering::Less    => { i += 1; }
                std::cmp::Ordering::Greater => { j += 1; }
            }
        }
        common as f64 / K as f64
    }
}
