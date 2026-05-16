// WeightMatrix — the unified object W.
//
// W(i,j) simultaneously encodes:
//   • topology  : coupling strength between field dimensions i and j
//   • memory    : Hopfield synaptic weight storing attractor patterns
//   • routing   : ∇W points toward the nearest attractor (used by SGR)
//
// Invariants (enforced after every mutation):
//   • Symmetric  : W[i][j] = W[j][i]
//   • Zero diag  : W[i][i] = 0
//
// Storage: flat row-major Vec<f32> of length D×D — heap-allocated, 256 KB per node.

use crate::{
    field::{FieldVec, zero_field},
    types::{D, ETA, LAMBDA, ENERGY_PER_FLOP},
};

#[derive(Clone)]
pub struct WeightMatrix {
    data: Vec<f32>,
}

impl WeightMatrix {
    pub fn zeros() -> Self {
        Self { data: vec![0.0f32; D * D] }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * D + j]
    }

    #[inline]
    fn raw_set(&mut self, i: usize, j: usize, v: f32) {
        self.data[i * D + j] = v;
    }

    // ── Axiom G / replication storage ───────────────────────────────────────
    // ΔW(a,b) += scale · ξ(a) · ξ(b)   with  scale = 1 / (k · d)
    pub fn hebbian_store(&mut self, xi: &[f32; D], scale: f32) -> f64 {
        for i in 0..D {
            for j in (i + 1)..D {
                let delta = scale * xi[i] * xi[j];
                self.data[i * D + j] += delta;
                self.data[j * D + i] += delta;
            }
        }
        // energy cost: D² multiply-adds
        (D * D) as f64 * ENERGY_PER_FLOP
    }

    // ── Axiom C: topological plasticity ─────────────────────────────────────
    // ΔW = η · (Ψᵢ ⊗ Ψⱼ) − λ · W
    pub fn plasticity_step(&mut self, psi: &[f32; D], eta: f32, lambda: f32) -> f64 {
        for i in 0..D {
            for j in (i + 1)..D {
                let idx_ij = i * D + j;
                let idx_ji = j * D + i;
                let delta = eta * psi[i] * psi[j] - lambda * self.data[idx_ij];
                self.data[idx_ij] += delta;
                self.data[idx_ji] += delta;
            }
        }
        (D * D) as f64 * ENERGY_PER_FLOP
    }

    /// Convenience wrapper using global ETA / LAMBDA constants.
    pub fn hebbian_update(&mut self, psi: &[f32; D]) -> f64 {
        self.plasticity_step(psi, ETA, LAMBDA)
    }

    // ── Invariant maintenance ────────────────────────────────────────────────
    pub fn enforce_symmetry(&mut self) {
        for i in 0..D {
            self.data[i * D + i] = 0.0;
            for j in (i + 1)..D {
                let avg = (self.data[i * D + j] + self.data[j * D + i]) * 0.5;
                self.data[i * D + j] = avg;
                self.data[j * D + i] = avg;
            }
        }
    }

    // ── Linear algebra ──────────────────────────────────────────────────────
    /// W · v
    pub fn matvec(&self, v: &[f32; D]) -> FieldVec {
        let mut out = zero_field();
        for i in 0..D {
            let mut s = 0.0f32;
            for j in 0..D {
                s += self.data[i * D + j] * v[j];
            }
            out[i] = s;
        }
        out
    }

    // ── Energy (classical Hopfield, normalised) ──────────────────────────────
    // E = −½ Ψᵀ W Ψ / d
    // Lower E → closer to a stored attractor.
    pub fn energy(&self, psi: &[f32; D]) -> f64 {
        let wv = self.matvec(psi);
        let raw: f64 = psi.iter().zip(wv.iter())
            .map(|(&p, &wp)| p as f64 * wp as f64)
            .sum();
        -0.5 * raw / D as f64
    }

    // ── Hopfield dynamics (Axiom E) ──────────────────────────────────────────
    // Single synchronous update: Ψ_new = sign(W · Ψ_old).
    // Guaranteed to decrease or maintain classical Hopfield energy.
    pub fn hopfield_step(&self, psi: &[f32; D]) -> FieldVec {
        let wv = self.matvec(psi);
        let mut out = zero_field();
        for (i, &x) in wv.iter().enumerate() {
            out[i] = if x >= 0.0 { 1.0 } else { -1.0 };
        }
        out
    }

    /// Iterate hopfield_step until convergence or `max_steps`.
    pub fn hopfield_relax(&self, psi: &[f32; D], max_steps: usize) -> FieldVec {
        let mut cur: FieldVec = {
            let mut c = zero_field();
            c.copy_from_slice(psi);
            c
        };
        for _ in 0..max_steps {
            let nxt = self.hopfield_step(&cur);
            if nxt[..] == cur[..] {
                return nxt;
            }
            cur = nxt;
        }
        cur
    }

    // ── W-CRDT merge (confidence-weighted average) ───────────────────────────
    // Used when two partitioned nodes re-join (Open Question II.3).
    pub fn merge(&self, other: &WeightMatrix, self_weight: f64, other_weight: f64) -> WeightMatrix {
        let total = self_weight + other_weight;
        let wa = (self_weight / total) as f32;
        let wb = (other_weight / total) as f32;
        let mut merged = WeightMatrix::zeros();
        for (i, (&a, &b)) in self.data.iter().zip(other.data.iter()).enumerate() {
            merged.data[i] = wa * a + wb * b;
        }
        merged.enforce_symmetry();
        merged
    }

    /// Frobenius cosine similarity — measures topology coupling between two nodes.
    pub fn coupling(&self, other: &WeightMatrix) -> f64 {
        let dot: f64 = self.data.iter().zip(other.data.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let na: f64 = self.data.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
        let nb: f64 = other.data.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 { return 0.0; }
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    pub fn as_slice(&self) -> &[f32] { &self.data }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use crate::field::bipolar_field;

    fn seeded_rng(seed: u64) -> StdRng { StdRng::seed_from_u64(seed) }

    #[test]
    fn zeros_energy_is_zero() {
        let w = WeightMatrix::zeros();
        let mut rng = seeded_rng(0);
        let psi = bipolar_field(&mut rng);
        assert_eq!(w.energy(&psi), 0.0);
    }

    #[test]
    fn symmetry_invariant_after_hebbian_store() {
        let mut w = WeightMatrix::zeros();
        let mut rng = seeded_rng(1);
        for _ in 0..10 {
            let xi = bipolar_field(&mut rng);
            w.hebbian_store(&xi, 1.0 / (3.0 * D as f32));
        }
        w.enforce_symmetry();
        for i in 0..D {
            for j in 0..D {
                let diff = (w.get(i, j) - w.get(j, i)).abs();
                assert!(diff < 1e-6, "W[{i},{j}]={} ≠ W[{j},{i}]={}", w.get(i,j), w.get(j,i));
            }
        }
    }

    #[test]
    fn zero_diagonal_invariant() {
        let mut w = WeightMatrix::zeros();
        let mut rng = seeded_rng(2);
        for _ in 0..5 {
            let xi = bipolar_field(&mut rng);
            w.hebbian_store(&xi, 1.0 / (3.0 * D as f32));
        }
        w.enforce_symmetry();
        for i in 0..D {
            assert_eq!(w.get(i, i), 0.0, "diagonal W[{i},{i}] ≠ 0");
        }
    }

    #[test]
    fn hopfield_step_decreases_or_maintains_energy() {
        let mut w = WeightMatrix::zeros();
        let mut rng = seeded_rng(3);
        for _ in 0..8 {
            let xi = bipolar_field(&mut rng);
            w.hebbian_store(&xi, 1.0 / (3.0 * D as f32));
        }
        w.enforce_symmetry();

        let query = bipolar_field(&mut rng);
        let e0 = w.energy(&query);
        let next = w.hopfield_step(&query);
        let e1 = w.energy(&next);
        assert!(e1 <= e0 + 1e-9, "energy increased: {e0} → {e1}");
    }

    #[test]
    fn stored_pattern_is_attractor() {
        let mut w = WeightMatrix::zeros();
        let mut rng = seeded_rng(4);
        let pattern = bipolar_field(&mut rng);
        w.hebbian_store(&pattern, 1.0 / D as f32);
        w.enforce_symmetry();

        let relaxed = w.hopfield_relax(&pattern, 20);
        // The stored pattern should be a fixed point (or very close)
        let sim = crate::field::bit_similarity(&pattern, &relaxed);
        assert!(sim >= 0.85, "pattern not an attractor: similarity={sim:.3}");
    }

    #[test]
    fn merge_preserves_symmetry() {
        let mut rng = seeded_rng(5);
        let mut wa = WeightMatrix::zeros();
        let mut wb = WeightMatrix::zeros();
        for _ in 0..5 {
            wa.hebbian_store(&bipolar_field(&mut rng), 0.01);
            wb.hebbian_store(&bipolar_field(&mut rng), 0.01);
        }
        wa.enforce_symmetry();
        wb.enforce_symmetry();
        let merged = wa.merge(&wb, 1.0, 1.0);
        for i in 0..D {
            for j in 0..D {
                let diff = (merged.get(i, j) - merged.get(j, i)).abs();
                assert!(diff < 1e-6);
            }
        }
    }
}
