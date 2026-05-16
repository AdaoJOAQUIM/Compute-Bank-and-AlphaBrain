pub mod error;
pub mod field;
pub mod pheromone;
pub mod types;
pub mod weight;

pub use error::{FtcError, Result};
pub use field::{FieldVec, bipolar_field, bit_similarity, binarise, cosine_similarity,
                dot, noisy_field, zero_field};
pub use pheromone::Pheromone;
pub use types::{
    NodeId, Energy, D, ETA, LAMBDA, BETA, K_B, T_ROOM, LANDAUER_MIN, ENERGY_PER_FLOP,
};
pub use weight::WeightMatrix;
