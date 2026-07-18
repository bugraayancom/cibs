//! Backend-agnostic tensor surface.
//!
//! Today the project targets Apple Silicon via MLX (`Array`). Encoder /
//! decoder code imports [`Array`] from this module (or `backend_mlx`) so a
//! future `tch` backend can swap the concrete type behind the same names.

#[cfg(feature = "mlx")]
pub use crate::backend_mlx::{eval, Array, Dtype};

#[cfg(not(any(feature = "mlx", feature = "tch")))]
compile_error!("enable one backend feature: `mlx` (Apple Silicon) or `tch` (reserved)");
