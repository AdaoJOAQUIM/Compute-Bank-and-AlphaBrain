// Field Theory of Computation — Research Runner
//
// Answers the four pre-code questions, then runs E1–E10 (Phase I)
// followed by the Phase II systematic experiments (II.1–II.3 + unification).
//
// Usage:  cargo run --bin run-emergence-tests

mod emergence;
mod phase2;

use ftc_findings::{FtcFindings, Phase2Findings};
use emergence::*;
use phase2::*;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!(" Field Theory of Computation — v4.1 Research Session");
    println!(" Two open problems · Seven axioms · Ten emergence tests");
    println!("═══════════════════════════════════════════════════════════\n");

    // ── Pre-code questions (Q1–Q4) ───────────────────────────────────────────
    println!("── Pre-code research questions ──────────────────────────────\n");

    println!("Q1: Has anyone unified distributed compute topology and memory\n\
              in a single mathematical object?\n");
    println!("  No satisfying prior unification exists. Hopfield networks (1982)\n\
              unify memory with attractor dynamics but are not a distributed compute\n\
              substrate. Reservoir computing encodes topology in recurrent state but\n\
              has no persistent associative memory. Ant-colony optimisation uses\n\
              pheromone fields for routing but not semantic memory retrieval. The\n\
              W-matrix unification proposed here — where W(i,j) simultaneously\n\
              encodes topology coupling, Hopfield synaptic weight, and SGR routing\n\
              gradient — has no direct precedent in the literature.\n");

    println!("Q2: Why has no one built a distributed Modern Hopfield system\n\
              for general compute?\n");
    println!("  The specific obstacle: distributed semantic retrieval without a\n\
              directory. Per-node capacity is already effectively infinite with\n\
              Modern Hopfield (exp(d/2) ≈ 10^55 patterns for d=256). The gap is\n\
              that finding WHICH node holds the target attractor basin requires\n\
              either (a) a directory [O(N) coordination], (b) broadcast [O(N)\n\
              messages], or (c) exact-key hashing [incompatible with partial-match\n\
              queries]. SGR is the proposed solution: gradient descent in the W-field\n\
              reaches the nearest replica in O(log N) hops without any directory.\n");

    println!("Q3: Does gradient descent in a distributed W-field converge to\n\
              the nearest attractor in O(log N) hops with high probability?\n");
    println!("  Partial proof sketch: with k = O(log N) replicas distributed\n\
              uniformly at random in an Erdős–Rényi G(N, p) graph (diameter O(log N)),\n\
              the expected number of hops before reaching a node whose W matrix has\n\
              seen the target pattern is bounded by the graph diameter × (1/overlap).\n\
              For similarity s = ⟨ξ_q, ξ⟩ ≥ 0.8, empirical data below either\n\
              confirms or refutes the conjecture H = O(log N / s²). This is\n\
              Open Question II.1 — answered experimentally by E7.\n");

    println!("Q4: How does a Hebbian weight matrix eliminate bookkeeping overhead?\n");
    println!("  Classical distributed storage (Redis, Cassandra) wastes energy on:\n\
              explicit indices, replication logs, TTL tracking, consistency protocols.\n\
              A Hebbian W matrix eliminates ALL of these because the address IS the\n\
              pattern (content-addressable: no index structure needed), replication\n\
              is implicit in the Hebbian update (each processing node auto-stores),\n\
              consistency is maintained by energy minimisation (W-CRDT merge), and\n\
              'eviction' is graceful decay (−λW term) not bookkeeping. The only\n\
              energy cost is the W update: ΔW = (1/k·d)·ξ⊗ξ — which IS the\n\
              Landauer-minimum encoding of the pattern's information. Bookkeeping\n\
              cost is exactly zero because W is simultaneously the index and the data.\n");

    println!("────────────────────────────────────────────────────────────\n");

    // ── Run all emergence tests ───────────────────────────────────────────────
    let mut f = FtcFindings::new();

    println!("Running E1  — Topology beats oracle …");
    run_e1(&mut f);
    print_result("E1", f.e1_topology_beats_oracle,
        &format!("SGR {:.2} hops vs oracle {:.2} hops",
                 f.e1_sgr_mean_hops, f.e1_oracle_mean_hops));

    println!("Running E2  — Half-kill resilience …");
    run_e2(&mut f);
    print_result("E2", f.e2_half_kill_resilience,
        &format!("throughput after 50% kill: {:.1}%",
                 f.e2_throughput_after_kill * 100.0));

    println!("Running E3  — Spontaneous specialisation …");
    run_e3(&mut f);
    print_result("E3", f.e3_spontaneous_specialise,
        &format!("entropy reduction: {:.1}%", f.e3_entropy_reduction_pct));

    println!("Running E4  — Kolmogorov compression …");
    run_e4(&mut f);
    print_result("E4", f.e4_kolmogorov_compress,
        &format!("compressibility gain: {:.1}%", f.e4_compressibility_gain_pct));

    println!("Running E5  — Cross-provider attractor …");
    run_e5(&mut f);
    print_result("E5", f.e5_cross_group_attractor,
        "SGR path crossed group boundary");

    println!("Running E6  — Surprise requirement …");
    run_e6(&mut f);
    print_result("E6", f.e6_emergent_behavior_count >= 3,
        &format!("{} emergent behaviours catalogued",
                 f.e6_emergent_behavior_count));

    println!("Running E7  — Distributed capacity exceeds single node …");
    run_e7(&mut f);
    print_result("E7", f.e7_distributed_capacity,
        &format!("mean fidelity: {:.3}  |  {}",
                 f.e7_mean_retrieval_fidelity, f.ii1_sgr_depth_formula));

    println!("Running E8  — Holographic survival after 75% node loss …");
    run_e8(&mut f);
    print_result("E8", f.e8_holographic_survival,
        &format!("post-kill fidelity: {:.3}", f.e8_fidelity_after_75pct_loss));

    println!("Running E9  — Spontaneous consolidation …");
    run_e9(&mut f);
    print_result("E9", f.e9_consolidation,
        &format!("top-10 basin ratio: {:.3}  |  bot-10 basin ratio: {:.3}",
                 f.e9_basin_top10_ratio, f.e9_basin_bot10_ratio));

    println!("Running E10 — Energy approaches Landauer bound …");
    run_e10(&mut f);
    print_result("E10", f.e10_landauer_approach,
        &format!("Landauer ratio: early={:.2e}  final={:.2e}",
                 f.e10_landauer_ratio_early, f.e10_landauer_ratio_final));

    println!("Running II.3 — W-CRDT preserves attractor geometry …");
    run_ii3(&mut f);
    print_result("II.3", f.ii3_crdt_preserves_attractors,
        "merged W still recovers ξ as a stable attractor");

    println!("Running unification metric …");
    run_unification_metric(&mut f);
    println!("  Unification: path-overlap={:.3}  verdict={}",
             f.memory_topology_correlation, f.unified_or_separate);

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n════════════════════════════════════════════════════════════");
    println!(" RESULTS SUMMARY");
    println!("════════════════════════════════════════════════════════════");
    println!(" Passed {}/10 emergence tests", f.passed_count());
    println!(" SGR conjecture H=O(log N/s²): {}",
             if f.ii1_conjecture_holds { "HOLDS" } else { "REFUTED — need new mechanism" });
    println!(" W-CRDT attractor preservation: {}",
             if f.ii3_crdt_preserves_attractors { "YES" } else { "NO — basin geometry lost in merge" });
    println!(" Consolidation converges: {}",
             if f.ii2_consolidation_converges { "YES" } else { "NO — still oscillating" });
    println!(" Unification verdict: {}", f.unified_or_separate);

    println!("\n── Emergent behaviours observed ─────────────────────────────");
    for (i, eb) in f.emergent_behaviors.iter().enumerate() {
        println!("  EB{}: {}", i + 1, eb.name);
        println!("       {}", eb.description);
    }

    println!("\n── Open questions discovered during experiment ───────────────");
    for oq in &f.open_questions_discovered {
        println!("  {}: {}", oq.id, oq.question);
        if let Some(a) = &oq.partial_answer {
            println!("     → {a}");
        }
    }

    println!("\n── Load-bearing axioms ──────────────────────────────────────");
    for (test, axiom) in &f.load_bearing_axioms {
        println!("  {test:<35} ← Axiom {axiom}");
    }

    println!("\n── Landauer bound analysis ──────────────────────────────────");
    println!("  Theoretical minimum: {:.3e} J/bit", ftc_core::LANDAUER_MIN);
    println!("  Early ratio:  {:.3e}×  (above minimum — required by 2nd law)",
             f.e10_landauer_ratio_early);
    println!("  Final ratio:  {:.3e}×  ({})",
             f.e10_landauer_ratio_final,
             if f.e10_landauer_ratio_final < f.e10_landauer_ratio_early {
                 "improving — consolidation reducing overhead"
             } else {
                 "not yet improving — consolidation incomplete"
             });

    // Emit Phase I JSON
    let json = serde_json::to_string_pretty(&f).expect("serialisation failed");
    std::fs::write("ftc_findings.json", &json).expect("could not write ftc_findings.json");
    println!("\nFindings written to ftc_findings.json");

    // ── Phase II ─────────────────────────────────────────────────────────────
    println!("\n════════════════════════════════════════════════════════════");
    println!(" PHASE II — SYSTEMATIC OPEN QUESTION EXPERIMENTS");
    println!("════════════════════════════════════════════════════════════\n");

    let mut f2 = Phase2Findings::new();

    println!("Running II.1 — Routing depth H(N,d,s) for N∈{{5,10,20,50}} …");
    println!("  (measuring empirical curve vs conjecture H=O(log N/s²))");
    run_ii1_routing_depth(&mut f2);
    println!("  Conjecture holds: {}   violations: {}",
             if f2.ii1_conjecture_holds { "YES" } else { "NO" },
             f2.ii1_conjecture_violations);
    println!("  Empirical fit: {}", f2.ii1_empirical_formula);
    println!("  Routing table:");
    println!("  {:>5} {:>6} {:>10} {:>14} {:>8}",
             "N", "s", "mean_hops", "conjectured", "holds?");
    for pt in &f2.routing_points {
        println!("  {:>5} {:>6.2} {:>10.2} {:>14.2} {:>8}",
                 pt.n, pt.similarity, pt.mean_hops,
                 pt.conjectured_bound,
                 if pt.conjecture_holds { "✓" } else { "✗" });
    }

    println!("\nRunning II.2 — Consolidation Nash equilibrium (2000 ticks) …");
    println!("  (sampling basin radii every 100 ticks; CV convergence test)");
    run_ii2_consolidation(&mut f2);
    println!("  Converged (CV<0.15 in last 5 snapshots): {}",
             if f2.ii2_converged { "YES" } else { "NO" });
    println!("  Final CV: {:.4}", f2.ii2_final_cv);
    println!("  Top-10 basin ratio (final/initial): {:.3}", f2.ii2_top_radius_ratio);
    println!("  Bot-10 basin ratio (final/initial): {:.3}", f2.ii2_bot_radius_ratio);
    if f2.consolidation_ticks.len() >= 3 {
        println!("  Basin radius time-series (top / bot):");
        let step = f2.consolidation_ticks.len() / 5;
        for i in (0..f2.consolidation_ticks.len()).step_by(step.max(1)).take(6) {
            let t  = f2.consolidation_ticks[i];
            let rt = f2.consolidation_top_radii.get(i).copied().unwrap_or(0.0);
            let rb = f2.consolidation_bot_radii.get(i).copied().unwrap_or(0.0);
            println!("    tick {:>5}  top={:.3}  bot={:.3}", t, rt, rb);
        }
    }

    println!("\nRunning II.3 — W-CRDT stress test (10 patterns, partition+merge) …");
    run_ii3_crdt_stress(&mut f2);
    println!("  Patterns tested:   {}", f2.ii3_patterns_tested);
    println!("  Patterns survived (≥0.90 sim): {}/{}", f2.ii3_patterns_survived, f2.ii3_patterns_tested);
    println!("  Mean similarity after merge: {:.3}", f2.ii3_merge_mean_similarity);
    println!("  Min  similarity after merge: {:.3}", f2.ii3_merge_min_similarity);
    println!("  Survived mean |E|:  {:.4}   Failed mean |E|: {:.4}",
             f2.ii3_survived_mean_energy, f2.ii3_failed_mean_energy);
    print_result("II.3 robust", f2.ii3_crdt_robust,
        &format!("{}/{} patterns survive multi-merge",
                 f2.ii3_patterns_survived, f2.ii3_patterns_tested));

    println!("\nRunning Unification metric — 2000-tick deep measurement …");
    run_unification_2000(&mut f2, &mut f);
    println!("  Jaccard(SGR path, Hebbian history) after 2000 ticks: {:.3}",
             f2.unification_jaccard_2000);
    println!("  Mean W coupling across node pairs: {:.4}", f2.unification_mean_coupling);
    println!("  Unification verdict: {}", f2.unification_verdict_2000);

    // ── Phase II summary ──────────────────────────────────────────────────────
    println!("\n════════════════════════════════════════════════════════════");
    println!(" PHASE II SUMMARY");
    println!("════════════════════════════════════════════════════════════");
    println!(" II.1 H(N,d,s) conjecture: {}",
             if f2.ii1_conjecture_holds { "HOLDS — SGR is O(log N/s²)" }
             else { "PARTIAL — some (N,s) pairs exceed bound" });
    println!(" II.2 Nash convergence:    {}   (CV={:.4})",
             if f2.ii2_converged { "CONVERGED" } else { "OSCILLATING" },
             f2.ii2_final_cv);
    println!(" II.3 W-CRDT robustness:   {}   ({}/{} patterns)",
             if f2.ii3_crdt_robust { "ROBUST" } else { "FRAGILE" },
             f2.ii3_patterns_survived, f2.ii3_patterns_tested);
    println!(" Unification (2000t):      {}   (Jaccard={:.3})",
             f2.unification_verdict_2000, f2.unification_jaccard_2000);

    // Emit Phase II JSON
    let json2 = serde_json::to_string_pretty(&f2).expect("serialisation failed");
    std::fs::write("phase2_findings.json", &json2)
        .expect("could not write phase2_findings.json");
    println!("\nPhase II findings written to phase2_findings.json");
    println!("════════════════════════════════════════════════════════════\n");
}

fn print_result(test: &str, passed: bool, detail: &str) {
    let mark = if passed { "✓ PASS" } else { "✗ FAIL" };
    println!("  [{mark}] {test:<4}  {detail}");
}
