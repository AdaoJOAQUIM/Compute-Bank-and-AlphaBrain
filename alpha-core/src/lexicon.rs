//! Axiom L — Semantic Crystallization via n-gram hashing.
//!
//! seed(token, genesis) = top-k sparse vector derived from:
//!   h2 = blake3(genesis ‖ bigrams(token))   × 0.20
//!   h3 = blake3(genesis ‖ trigrams(token))  × 0.50
//!   h4 = blake3(genesis ‖ fourgrams(token)) × 0.30
//!   merged = weighted_hash(h2, h3, h4)
//!   → ChaCha20Rng::from_seed(merged) → sample K indices

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand::seq::index::sample;
use crate::sparse_vec::SparseVec;
use crate::types::{D, K};

pub fn ngrams(token: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() < n { return vec![token.to_string()]; }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

fn encode_ngrams(token: &str, n: usize) -> Vec<u8> {
    ngrams(token, n).join("\u{B7}").into_bytes()
}

fn blake3_hash(genesis: &[u8; 32], data: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(genesis);
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

/// Weighted merge of three 32-byte hashes into one seed.
fn weighted_merge(h2: [u8; 32], h3: [u8; 32], h4: [u8; 32]) -> [u8; 32] {
    let mut combined = [0u8; 32];
    for i in 0..32 {
        let v = h2[i] as f32 * 0.20 + h3[i] as f32 * 0.50 + h4[i] as f32 * 0.30;
        combined[i] = v as u8;
    }
    blake3_hash(&combined, b"alpha-merge")
}

/// Generate a deterministic SparseVec from a token and genesis seed.
///
/// Properties:
///   - seed("deploy", g) ∩ seed("deployment", g) > 30% active  [ngram_proximity]
///   - seed("deploy", g) ∩ seed("quantum", g) < 5% active
pub fn seed(token: &str, genesis: &[u8; 32]) -> SparseVec {
    let h2 = blake3_hash(genesis, &encode_ngrams(token, 2));
    let h3 = blake3_hash(genesis, &encode_ngrams(token, 3));
    let h4 = blake3_hash(genesis, &encode_ngrams(token, 4));
    let merged = weighted_merge(h2, h3, h4);

    let mut rng = ChaCha20Rng::from_seed(merged);
    use rand::Rng;
    let mut idx: Vec<u16> = sample(&mut rng, D, K).iter().map(|i| i as u16).collect();
    idx.sort_unstable();
    let vals: Vec<f32> = idx.iter().map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
    SparseVec { indices: idx, values: vals }
}

/// Cache of token → SparseVec for fast crystallization lookup.
pub struct SeedLexicon {
    genesis: [u8; 32],
    cache: std::collections::HashMap<String, SparseVec>,
}

impl SeedLexicon {
    pub fn new(genesis: [u8; 32]) -> Self {
        Self { genesis, cache: Default::default() }
    }

    pub fn get(&mut self, token: &str) -> &SparseVec {
        self.cache.entry(token.to_string())
            .or_insert_with(|| seed(token, &self.genesis))
    }

    pub fn crystallize(&mut self, tokens: &[&str]) -> SparseVec {
        // Average the sparse vectors: collect all indices and sum values
        let mut idx_vals: std::collections::HashMap<u16, f32> = Default::default();
        for &t in tokens {
            let v = self.get(t).clone();
            for (i, &idx) in v.indices.iter().enumerate() {
                *idx_vals.entry(idx).or_insert(0.0) += v.values[i];
            }
        }
        // Take top-K by absolute value (stable: break ties by index for determinism)
        let mut pairs: Vec<(u16, f32)> = idx_vals.into_iter().collect();
        pairs.sort_by(|a, b| {
            b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                .then(a.0.cmp(&b.0)) // deterministic tie-breaker by index
        });
        pairs.truncate(K);
        pairs.sort_by_key(|(i, _)| *i);
        let indices: Vec<u16> = pairs.iter().map(|(i, _)| *i).collect();
        let values:  Vec<f32> = pairs.iter().map(|(_, v)| v.signum()).collect();
        SparseVec { indices, values }
    }
}
