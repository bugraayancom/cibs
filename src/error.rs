use std::path::PathBuf;

/// Top-level error type for the qwen3-asr crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("config error: {0}")]
    Config(String),

    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("missing tensor in checkpoint: {0}")]
    MissingTensor(String),

    #[error("audio decode error: {0}")]
    Audio(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("generation error: {0}")]
    Generate(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

#[cfg(feature = "mlx")]
impl From<mlx_rs::error::Exception> for Error {
    fn from(e: mlx_rs::error::Exception) -> Self {
        Error::Backend(e.to_string())
    }
}

impl From<tokenizers::Error> for Error {
    fn from(e: tokenizers::Error) -> Self {
        Error::Tokenizer(e.to_string())
    }
}
