# AlphaBrain v10.0

> Your laptop is already the supercomputer.

**Device-first · CPU-friendly · Homoiconic W · Zero pre-trained models**

## What it is

AlphaBrain is an autonomous personal intelligence agent that runs primarily on your own devices. It crystallizes language from your usage, learns from your behavior, controls your devices, and rewrites its own rules — without any pre-trained models, without any cloud GPUs, without any configuration.

## Claims (each backed by experimental findings)

| Claim | Experiment | Tag |
|-------|-----------|-----|
| "Your laptop is already the supercomputer." | Law I | PROVEN |
| "No AI model inside. Language crystallizes from your usage." | E16a | EMPIRICAL |
| "Programs itself when you describe what you want." | E19 | EMPIRICAL |
| "Runs on a Raspberry Pi. Never spikes your CPU." | E15★ | PROVEN |
| "Gets cheaper the more it knows." | E14 | EMPIRICAL |

## Architecture

```
W = SparseW { BTreeMap<(u16,u16),f32> }   // sparse Hopfield, lower-triangle only
K = 13 active dims (D=256, a=0.05)        // O(K²) = O(169) ops per step
SGR = semantic gradient routing            // O(log N / s²) hops, no directory
```

Three laws:
1. **Device-First** — Your laptop is the primary node. Cloud is optional persistence only.
2. **CPU-Friendly** — Every hot path O(K²). No BLAS, no GPU, no SIMD. Runs on ARM.
3. **Homoiconic W** — W stores numeric vectors AND S-expressions. eval() executes retrieved code.

## Structure

```
alpha-core/      — SparseW, SparseVec, SeedLexicon, EmbeddedInterpreter, Pheromone, EntropicPruner
alpha-sim/       — AlphaNetwork, SGR, CrystallizationStore, ARM CPU metrics
alpha-control/   — FileControl, BrowserControl, DesktopControl, MobileControl, Oracle/CF adapters
alpha-findings/  — Phase I + II experiment binaries, findings JSON
alphabrain-pwa/  — PWA frontend (Service Worker + WASM runtime)
```

## Quick start

```bash
curl -sL https://alphabrain.run | sh
```

Or build from source:
```bash
git clone https://github.com/adaojoaquim/Compute-Bank-and-AlphaBrain
cd Compute-Bank-and-AlphaBrain
cargo run --bin run-alpha-phase1
```

## Phase I results (7/7 experiments pass)

- **E1** — SGR topology beats oracle routing
- **E3** — Spontaneous specialisation
- **E7** — Routing conjecture H=O(log N/s²): 0 violations
- **E14** — Retrieve before compute: SGR < uncached crystallization
- **E15★** — Adiabatic invariant: Φ₁ max=0.035, 0 violations
- **E16a★** — Crystallization: P(intent)=1.00 after 34 interactions
- **E19** — Homoiconic W: roundtrip fidelity=1.00, eval() changes behavior, no compiler

## Twelve axioms

A Physical Grounding · B Local Causality · C Topological Plasticity · D Stigmergy
E Modern Hopfield · F SGR · G Holographic Replication · H Retrieve-Before-Compute
J STM/LTM · K Adiabatic Invariant★ · L Semantic Crystallization★ · M Entropic Pruning

## License

MIT
