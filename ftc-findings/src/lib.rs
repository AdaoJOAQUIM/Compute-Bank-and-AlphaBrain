use serde::{Deserialize, Serialize};

// ── Phase II data structures ──────────────────────────────────────────────────

/// One data point in the H(N, d, s) routing-depth experiment.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RoutingPoint {
    pub n: usize,
    pub similarity: f64,
    pub mean_hops: f64,
    pub mean_fidelity: f64,
    pub log_n: f64,
    pub conjectured_bound: f64,
    pub conjecture_holds: bool,
}

/// All Phase II findings in one struct (serialised to phase2_findings.json).
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Phase2Findings {
    // II.1 — routing depth
    pub routing_points:           Vec<RoutingPoint>,
    pub ii1_conjecture_holds:     bool,
    pub ii1_conjecture_violations: usize,
    pub ii1_fit_coefficient:      f64,
    pub ii1_fit_r_squared:        f64,
    pub ii1_empirical_formula:    String,

    // II.2 — consolidation Nash
    pub ii2_converged:            bool,
    pub ii2_final_cv:             f64,
    pub ii2_top_radius_ratio:     f64,
    pub ii2_bot_radius_ratio:     f64,
    pub ii2_ticks_to_observe:     usize,
    pub consolidation_ticks:      Vec<u64>,
    pub consolidation_top_radii:  Vec<f64>,
    pub consolidation_bot_radii:  Vec<f64>,

    // II.3 — W-CRDT stress
    pub ii3_patterns_tested:      usize,
    pub ii3_patterns_survived:    usize,
    pub ii3_merge_mean_similarity: f64,
    pub ii3_merge_min_similarity:  f64,
    pub ii3_crdt_robust:           bool,
    pub ii3_survived_mean_energy:  f64,
    pub ii3_failed_mean_energy:    f64,

    // Unification
    pub unification_jaccard_2000:  f64,
    pub unification_mean_coupling: f64,
    pub unification_verdict_2000:  String,
}

/// A behaviour that was not explicitly programmed but emerged from the
/// seven axioms during simulation.  ≥3 required to pass E6.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmergentBehavior {
    pub name: String,
    pub description: String,
    pub observed_at_tick: Option<u64>,
}

/// An open question uncovered during the experiment — beyond the three
/// pre-stated ones in the brief.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenQuestion {
    pub id: String,
    pub question: String,
    pub partial_answer: Option<String>,
}

/// The complete research output: one value per research question.
/// All fields are filled by running the emergence tests.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct FtcFindings {
    // ── Problem I — Topology ────────────────────────────────────────────────
    pub e1_topology_beats_oracle:    bool,
    pub e1_sgr_mean_hops:            f64,
    pub e1_oracle_mean_hops:         f64,
    pub e2_half_kill_resilience:     bool,
    pub e2_throughput_after_kill:    f64,
    pub e3_spontaneous_specialise:   bool,
    pub e3_entropy_reduction_pct:    f64,
    pub e4_kolmogorov_compress:      bool,
    pub e4_compressibility_gain_pct: f64,
    pub e5_cross_group_attractor:    bool,
    pub e6_emergent_behavior_count:  usize,

    // ── Problem II — Memory ─────────────────────────────────────────────────
    pub e7_distributed_capacity:     bool,
    pub e7_mean_retrieval_fidelity:  f64,
    pub e8_holographic_survival:     bool,
    pub e8_fidelity_after_75pct_loss: f64,
    pub e9_consolidation:            bool,
    pub e9_basin_top10_ratio:        f64,  // basin_top / basin_t0
    pub e9_basin_bot10_ratio:        f64,  // basin_bot / basin_t0
    pub e10_landauer_approach:       bool,
    pub e10_landauer_ratio_early:    f64,
    pub e10_landauer_ratio_final:    f64,

    // ── Open Questions (II.1–II.3) ──────────────────────────────────────────
    pub ii1_sgr_depth_formula:       String, // empirical bound H(N,d,s)
    pub ii1_conjecture_holds:        bool,   // H = O(log N / s²) ?
    pub ii2_consolidation_converges: bool,
    pub ii3_crdt_preserves_attractors: bool,

    // ── Unification ─────────────────────────────────────────────────────────
    pub memory_topology_correlation: f64,   // Pearson ρ(SGR path ↔ Hebbian path)
    pub unified_or_separate:         String, // "unified" | "interfering" | "independent"

    // ── Surprises ───────────────────────────────────────────────────────────
    pub emergent_behaviors:           Vec<EmergentBehavior>,
    pub open_questions_discovered:    Vec<OpenQuestion>,

    // ── Which axiom is load-bearing? ────────────────────────────────────────
    /// Maps test name → axiom letter whose removal breaks the test.
    pub load_bearing_axioms:          Vec<(String, String)>,
}

impl Phase2Findings {
    pub fn new() -> Self { Self::default() }
}

impl FtcFindings {
    pub fn new() -> Self { Self::default() }

    pub fn passed_count(&self) -> usize {
        [
            self.e1_topology_beats_oracle,
            self.e2_half_kill_resilience,
            self.e3_spontaneous_specialise,
            self.e4_kolmogorov_compress,
            self.e5_cross_group_attractor,
            self.e6_emergent_behavior_count >= 3,
            self.e7_distributed_capacity,
            self.e8_holographic_survival,
            self.e9_consolidation,
            self.e10_landauer_approach,
        ].iter().filter(|&&b| b).count()
    }
}
