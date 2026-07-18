//! MLX backend: type aliases and small helpers shared by the model code.
//!
//! The model code (encoder/decoder) is written directly against
//! [`mlx_rs::Array`]; this module centralizes the few backend-specific
//! helpers so a future second backend has a single seam to replace.

pub use mlx_rs::Array;
pub use mlx_rs::Dtype;

use crate::error::Result;

/// Force evaluation of a lazy MLX array (MLX builds graphs lazily; without
/// periodic eval the graph grows unboundedly during autoregressive decode).
pub fn eval(arrays: &[&Array]) -> Result<()> {
    for a in arrays {
        a.eval()?;
    }
    Ok(())
}
