use crate::types::NodeId;

#[derive(Debug)]
pub enum FtcError {
    NodeNotFound(NodeId),
    InvalidPattern(String),
    RoutingFailed { hops: usize, reason: &'static str },
    NetworkTooSmall { required: usize, actual: usize },
    MathError(&'static str),
}

impl std::fmt::Display for FtcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FtcError::NodeNotFound(id) => write!(f, "node {} not found", id),
            FtcError::InvalidPattern(s) => write!(f, "invalid pattern: {s}"),
            FtcError::RoutingFailed { hops, reason } => {
                write!(f, "routing failed after {hops} hops: {reason}")
            }
            FtcError::NetworkTooSmall { required, actual } => {
                write!(f, "network too small: need {required} nodes, have {actual}")
            }
            FtcError::MathError(s) => write!(f, "math error: {s}"),
        }
    }
}

impl std::error::Error for FtcError {}

pub type Result<T> = std::result::Result<T, FtcError>;
