pub mod audio;
pub mod config;
pub mod error;
pub mod mel;
pub mod tensor;
pub mod tokenizer;

#[cfg(feature = "mlx")]
pub mod backend_mlx;
#[cfg(feature = "mlx")]
pub mod decoder;
#[cfg(feature = "mlx")]
pub mod encoder;
#[cfg(feature = "mlx")]
pub mod generate;
#[cfg(feature = "mlx")]
pub mod weights;

#[cfg(feature = "mlx")]
pub mod asr;
#[cfg(feature = "mlx")]
pub mod translate;

#[cfg(feature = "tch")]
pub mod backend_tch;

#[cfg(feature = "tch")]
compile_error!(
    "the `tch` (libtorch) backend is not implemented yet; build with `--features mlx` instead"
);

pub use config::ModelConfig;
pub use error::{Error, Result};
