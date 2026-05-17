//! Axiom L — Semantic Crystallization.
//! Maps natural language tokens to SparseVec attractors via n-gram seeds.

use alpha_core::{SparseVec, SeedLexicon};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentRecord {
    pub tokens:    Vec<String>,
    pub vector:    SparseVec,
    pub count:     usize,
    pub confirmed: bool,
}

pub struct CrystallizationStore {
    lexicon: SeedLexicon,
    pub intents: Vec<IntentRecord>,
    pub interactions: usize,
}

impl CrystallizationStore {
    pub fn new(genesis: [u8; 32]) -> Self {
        Self { lexicon: SeedLexicon::new(genesis), intents: Vec::new(), interactions: 0 }
    }

    /// Record one NL input and crystallize.
    pub fn observe(&mut self, tokens: &[&str]) -> SparseVec {
        self.interactions += 1;
        let vector = self.lexicon.crystallize(tokens);
        let token_strs: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
        // Find existing intent with similar vector, or create new
        let mut matched_idx = None;
        for (i, rec) in self.intents.iter().enumerate() {
            if rec.vector.bit_similarity(&vector) > 0.5 {
                matched_idx = Some(i);
                break;
            }
        }
        if let Some(idx) = matched_idx {
            self.intents[idx].count += 1;
            if self.intents[idx].count >= 5 {
                self.intents[idx].confirmed = true;
            }
        } else {
            self.intents.push(IntentRecord {
                tokens: token_strs,
                vector: vector.clone(),
                count: 1,
                confirmed: false,
            });
        }
        vector
    }

    /// P(intent_correct) after N interactions — estimated from confirmed intents.
    pub fn intent_accuracy(&self) -> f64 {
        let confirmed = self.intents.iter().filter(|r| r.confirmed).count();
        if self.intents.is_empty() { 0.0 } else { confirmed as f64 / self.intents.len() as f64 }
    }

    pub fn n_star_estimate(&self) -> usize {
        let target_intents = (self.intents.len() as f64 * 0.90).ceil() as usize;
        let confirmed = self.intents.iter().filter(|r| r.confirmed).count();
        let remaining = target_intents.saturating_sub(confirmed);
        self.interactions + remaining * 5
    }
}

/// Crystallize tokens into a SparseVec (stateless helper).
pub fn crystallize_intent(tokens: &[&str], lexicon: &mut SeedLexicon) -> SparseVec {
    lexicon.crystallize(tokens)
}
