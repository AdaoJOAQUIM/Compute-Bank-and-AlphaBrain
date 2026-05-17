use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct AlphaFindings {
    // Q1-Q8 research questions
    pub q1_prior_art_gap: String,
    pub q2_crystallization_n_star: usize,
    pub q2_cpu_us_per_step: f64,
    pub q2_viable_on_raspberry_pi: bool,
    pub q3_phi1_universal_formula: String,
    pub q4_click_bits: f64,
    pub q4_dwell_bits: f64,
    pub q4_correction_bits: f64,
    pub q4_n_behavioral_star: usize,
    pub q5_delta_stability_bound: String,
    pub q5_lean4_formalizable: bool,
    pub q6_n_star_macbook: usize,
    pub q6_w_star_documents: usize,
    pub q6_days_to_threshold: f64,
    pub q7_recommended_stack: String,
    pub q8a_eval_vs_compile_ratio: f64,
    pub q8b_three_use_cases: Vec<String>,
    pub q8c_interpreter_choice: String,
    pub q8_homoiconic_verdict: String,

    // Emergence test results
    pub e1_topology_beats_oracle: bool,
    pub e1_sgr_mean_hops: f64,
    pub e1_oracle_mean_hops: f64,

    pub e3_specialisation: bool,
    pub e3_entropy_reduction_pct: f64,

    pub e7_conjecture_holds: bool,
    pub e7_violations: usize,

    pub e14_retrieve_beats_compute: bool,
    pub e14_second_cpu_fraction: f64,

    pub e15_adiabatic_ok: bool,
    pub e15_phi1_max_under_spike: f64,
    pub e15_violations: usize,

    pub e16a_intent_accuracy: f64,
    pub e16a_n_interactions: usize,
    pub e16a_passes: bool,

    pub e19_sexpr_roundtrip_ok: bool,
    pub e19_eval_changed_behavior: bool,
    pub e19_compilation_needed: bool,
    pub e19_fuel_cost: u64,

    // Epistemic tags
    pub primary_breakthrough: String,
}

impl AlphaFindings {
    pub fn new() -> Self { Self::default() }
}
