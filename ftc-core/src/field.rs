// FieldVec: a d-dimensional bipolar state vector  Ψ ∈ {−1,+1}^D.
// All arithmetic is f32 internally; energy sums are promoted to f64.

use crate::types::D;

/// Heap-allocated field vector (avoids 1 KB stack frames from [f32; 256]).
pub type FieldVec = Box<[f32; D]>;

/// All-zero field (unactivated state — not a valid Hopfield state, used as seed).
pub fn zero_field() -> FieldVec {
    Box::new([0.0f32; D])
}

/// Random bipolar field  Ψᵢ ∈ {−1, +1}.
pub fn bipolar_field(rng: &mut impl rand::Rng) -> FieldVec {
    let mut f = zero_field();
    for x in f.iter_mut() {
        *x = if rng.gen::<bool>() { 1.0 } else { -1.0 };
    }
    f
}

/// Copy `base` and flip each bit independently with probability `noise ∈ [0,1]`.
pub fn noisy_field(base: &[f32; D], noise: f64, rng: &mut impl rand::Rng) -> FieldVec {
    let mut f = zero_field();
    for (i, x) in f.iter_mut().enumerate() {
        *x = if rng.gen::<f64>() < noise { -base[i] } else { base[i] };
    }
    f
}

/// Inner product  ⟨a, b⟩  (f64 accumulator to avoid float32 drift).
pub fn dot(a: &[f32; D], b: &[f32; D]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum()
}

/// Cosine similarity in [−1, 1].
pub fn cosine_similarity(a: &[f32; D], b: &[f32; D]) -> f64 {
    let ab = dot(a, b);
    let na = a.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
    let nb = b.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();
    if na < 1e-10 || nb < 1e-10 { return 0.0; }
    (ab / (na * nb)).clamp(-1.0, 1.0)
}

/// Fraction of bits that agree (normalised Hamming similarity).
pub fn bit_similarity(a: &[f32; D], b: &[f32; D]) -> f64 {
    let agree = a.iter().zip(b.iter())
        .filter(|(&x, &y)| x.signum() == y.signum())
        .count();
    agree as f64 / D as f64
}

/// Hard-thresholded binarisation: sign of each element.
pub fn binarise(v: &[f32; D]) -> FieldVec {
    let mut out = zero_field();
    for (i, &x) in v.iter().enumerate() {
        out[i] = if x >= 0.0 { 1.0 } else { -1.0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn bipolar_values() {
        let mut rng = StdRng::seed_from_u64(42);
        let f = bipolar_field(&mut rng);
        for &x in f.iter() {
            assert!(x == 1.0 || x == -1.0);
        }
    }

    #[test]
    fn cosine_self_is_one() {
        let mut rng = StdRng::seed_from_u64(1);
        let f = bipolar_field(&mut rng);
        let s = cosine_similarity(&f, &f);
        assert!((s - 1.0).abs() < 1e-6, "cosine_similarity(f,f) = {s}");
    }

    #[test]
    fn noisy_field_flips_nothing_at_zero_noise() {
        let mut rng = StdRng::seed_from_u64(7);
        let base = bipolar_field(&mut rng);
        let noisy = noisy_field(&base, 0.0, &mut rng);
        assert_eq!(bit_similarity(&base, &noisy), 1.0);
    }
}
