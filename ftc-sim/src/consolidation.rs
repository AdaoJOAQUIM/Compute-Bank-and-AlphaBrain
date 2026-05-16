// Hebbian consolidation monitor — Open Question II.2.
//
// Tracks basin radii over time to detect Nash-equilibrium convergence.
// A "Nash equilibrium" here means: no node can unilaterally improve its
// local energy landscape by deviating from the Hebbian update rule.
// Empirically: basin radii stop changing (coefficient of variation < ε).

use ftc_core::{NodeId, D};
use crate::{network::Network, replication::measure_basin_radius};

/// Snapshot of basin radii for a set of tracked patterns at one tick.
#[derive(Clone, Debug)]
pub struct BasinSnapshot {
    pub tick: u64,
    /// (pattern_index, basin_radius_at_node_0)
    pub radii: Vec<(usize, f64)>,
}

/// Running history of basin-radius snapshots.
pub struct ConsolidationTracker {
    pub snapshots: Vec<BasinSnapshot>,
    /// Which pattern indices are being tracked.
    pub tracked_indices: Vec<usize>,
    /// Sample interval in ticks.
    pub interval: u64,
}

impl ConsolidationTracker {
    pub fn new(tracked_indices: Vec<usize>, interval: u64) -> Self {
        Self { snapshots: Vec::new(), tracked_indices, interval }
    }

    /// Record a snapshot if `tick` is a multiple of `interval`.
    pub fn maybe_record(
        &mut self,
        tick: u64,
        patterns: &[Box<[f32; D]>],
        node_id: NodeId,
        network: &Network,
        rng: &mut impl rand::Rng,
    ) {
        if tick % self.interval != 0 { return; }
        let radii: Vec<(usize, f64)> = self.tracked_indices.iter().map(|&i| {
            let r = measure_basin_radius(&patterns[i], node_id, network, rng, 12, 0.80);
            (i, r)
        }).collect();
        self.snapshots.push(BasinSnapshot { tick, radii });
    }

    /// Coefficient of variation of basin radii in the last `window` snapshots,
    /// averaged across tracked patterns.  Low CV = system converged.
    pub fn convergence_cv(&self, window: usize) -> f64 {
        let n = self.snapshots.len();
        if n < 2 { return f64::INFINITY; }
        let recent = &self.snapshots[n.saturating_sub(window)..];

        let n_patterns = self.tracked_indices.len();
        if n_patterns == 0 { return 0.0; }

        let mut total_cv = 0.0f64;
        for pat_pos in 0..n_patterns {
            let series: Vec<f64> = recent.iter()
                .filter_map(|s| s.radii.get(pat_pos).map(|(_, r)| *r))
                .collect();
            if series.len() < 2 { continue; }
            let mean = series.iter().sum::<f64>() / series.len() as f64;
            if mean < 1e-9 { continue; }
            let variance = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (series.len() - 1) as f64;
            total_cv += variance.sqrt() / mean;
        }
        total_cv / n_patterns as f64
    }

    /// True when the last `window` snapshots show CV < threshold.
    pub fn has_converged(&self, window: usize, cv_threshold: f64) -> bool {
        self.convergence_cv(window) < cv_threshold
    }

    /// Mean basin radius of `indices` in the last snapshot.
    pub fn mean_radius_last(&self, indices: &[usize]) -> f64 {
        let snap = match self.snapshots.last() { Some(s) => s, None => return 0.0 };
        let vals: Vec<f64> = snap.radii.iter()
            .filter(|(i, _)| indices.contains(i))
            .map(|(_, r)| *r)
            .collect();
        if vals.is_empty() { return 0.0; }
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    /// Mean basin radius of `indices` in the first snapshot.
    pub fn mean_radius_first(&self, indices: &[usize]) -> f64 {
        let snap = match self.snapshots.first() { Some(s) => s, None => return 0.0 };
        let vals: Vec<f64> = snap.radii.iter()
            .filter(|(i, _)| indices.contains(i))
            .map(|(_, r)| *r)
            .collect();
        if vals.is_empty() { return 0.0; }
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}
