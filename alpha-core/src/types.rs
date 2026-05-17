pub const D: usize = 256;
pub const K: usize = 13;         // active dims = D * 0.05 ≈ 13
pub const ETA: f32  = 0.01;
pub const LAMBDA: f32 = 0.001;
pub const K_B: f64 = 1.380_649e-23;
pub const T_ROOM: f64 = 293.15;
pub const LANDAUER_MIN: f64 = K_B * T_ROOM * std::f64::consts::LN_2;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PatternId(pub u64);
