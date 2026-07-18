//! Linear / embedding layers that transparently support MLX 4/8-bit
//! quantized weights (`weight` packed in u32 + `scales` + `biases`, as
//! produced by mlx-lm and published under mlx-community).

use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype};

use crate::error::Result;
use crate::weights::Weights;

#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    pub group_size: i32,
    pub bits: i32,
}

/// A linear layer weight: either a dense `[out, in]` matrix or an MLX
/// quantized triple.
#[derive(Clone)]
pub enum Linear {
    Dense(Array),
    Quantized {
        w: Array,
        scales: Array,
        biases: Array,
        q: QuantParams,
    },
}

impl Linear {
    /// Take `base.weight` (+ `.scales`/`.biases` when present) from the
    /// checkpoint. `quant` carries the model-level quantization config; a
    /// layer is only treated as quantized when its scales tensor exists.
    pub fn take(weights: &mut Weights, base: &str, quant: Option<QuantParams>) -> Result<Self> {
        let scales_name = format!("{base}.scales");
        match quant {
            Some(q) if weights.contains(&scales_name) => Ok(Linear::Quantized {
                w: weights.take(&format!("{base}.weight"))?,
                scales: weights.take(&scales_name)?,
                biases: weights.take(&format!("{base}.biases"))?,
                q,
            }),
            _ => Ok(Linear::Dense(weights.take(&format!("{base}.weight"))?)),
        }
    }

    /// `y = x @ W^T` (HF Linear layout).
    pub fn apply(&self, x: &Array) -> Result<Array> {
        match self {
            Linear::Dense(w) => Ok(ops::matmul(x, w.transpose()?)?),
            Linear::Quantized {
                w,
                scales,
                biases,
                q,
            } => Ok(ops::quantized_matmul(
                x,
                w,
                scales,
                biases,
                true,
                q.group_size,
                q.bits,
            )?),
        }
    }
}

/// Token embedding table, optionally quantized. Also serves as the (tied)
/// LM head via [`Embedding::as_linear`].
#[derive(Clone)]
pub enum Embedding {
    Dense(Array),
    Quantized {
        w: Array,
        scales: Array,
        biases: Array,
        q: QuantParams,
    },
}

impl Embedding {
    pub fn take(weights: &mut Weights, base: &str, quant: Option<QuantParams>) -> Result<Self> {
        let scales_name = format!("{base}.scales");
        match quant {
            Some(q) if weights.contains(&scales_name) => Ok(Embedding::Quantized {
                w: weights.take(&format!("{base}.weight"))?,
                scales: weights.take(&scales_name)?,
                biases: weights.take(&format!("{base}.biases"))?,
                q,
            }),
            _ => Ok(Embedding::Dense(weights.take(&format!("{base}.weight"))?)),
        }
    }

    /// Row lookup: `[n]` ids → `[n, hidden]` embeddings (bf16).
    pub fn lookup(&self, idx: &Array) -> Result<Array> {
        let emb = match self {
            Embedding::Dense(w) => w.index(idx),
            Embedding::Quantized {
                w,
                scales,
                biases,
                q,
            } => {
                // Dequantize just the selected rows (mlx-lm QuantizedEmbedding).
                let w_rows = w.index(idx);
                let s_rows = scales.index(idx);
                let b_rows = biases.index(idx);
                ops::dequantize(&w_rows, &s_rows, &b_rows, q.group_size, q.bits)?
            }
        };
        Ok(emb.as_dtype(Dtype::Bfloat16)?)
    }

    /// Use the (tied) embedding table as the output projection.
    pub fn as_linear(&self) -> Linear {
        match self {
            Embedding::Dense(w) => Linear::Dense(w.clone()),
            Embedding::Quantized {
                w,
                scales,
                biases,
                q,
            } => Linear::Quantized {
                w: w.clone(),
                scales: scales.clone(),
                biases: biases.clone(),
                q: *q,
            },
        }
    }
}
