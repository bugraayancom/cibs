use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::indexing::{take_axis, IndexOp};
use mlx_rs::{ops, Array, Dtype};

use crate::config::ModelConfig;
use crate::error::Result;
use crate::mel::LogMelSpectrogram;
use crate::weights::Weights;

/// PyTorch nn.LayerNorm default epsilon.
const LN_EPS: f32 = 1e-5;

/// `y = x @ W^T (+ b)` with HF Linear weight layout `[out, in]`.
pub(crate) fn linear(x: &Array, w: &Array, b: Option<&Array>) -> Result<Array> {
    let y = ops::matmul(x, w.transpose()?)?;
    match b {
        Some(b) => Ok(ops::add(&y, b)?),
        None => Ok(y),
    }
}

/// Exact (erf-based) GELU, matching HF's "gelu" activation.
pub(crate) fn gelu(x: &Array) -> Result<Array> {
    let sqrt2 = Array::from_f32(std::f32::consts::SQRT_2);
    let inner = ops::erf(&ops::divide(x, &sqrt2)?)?;
    let one = Array::from_f32(1.0);
    let half = Array::from_f32(0.5);
    Ok(ops::multiply(
        &ops::multiply(x, &half)?,
        &ops::add(&inner, &one)?,
    )?)
}

struct EncoderLayer {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    out_w: Array,
    out_b: Array,
    attn_ln_w: Array,
    attn_ln_b: Array,
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
    final_ln_w: Array,
    final_ln_b: Array,
}

/// Qwen3-ASR audio tower: conv frontend + windowed-attention transformer
/// + output projector (proj1/proj2), producing LM-space audio embeddings.
pub struct AudioEncoder {
    conv1_w: Array,
    conv1_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    conv3_w: Array,
    conv3_b: Array,
    conv_out_w: Array,
    layers: Vec<EncoderLayer>,
    ln_post_w: Array,
    ln_post_b: Array,
    proj1_w: Array,
    proj1_b: Array,
    proj2_w: Array,
    proj2_b: Array,
    pos_embedding: Array,
    n_heads: i32,
    head_dim: i32,
    chunk_len: i32,
    window_tokens: i32,
}

impl AudioEncoder {
    pub fn load(weights: &mut Weights, cfg: &ModelConfig) -> Result<Self> {
        let p = "thinker.audio_tower";
        let a = &cfg.audio;

        // PyTorch conv weights are [O, I, kh, kw]; MLX wants [O, kh, kw, I].
        let conv = |w: Array| -> Result<Array> { Ok(w.transpose_axes(&[0, 2, 3, 1])?) };

        let mut layers = Vec::with_capacity(a.encoder_layers as usize);
        for i in 0..a.encoder_layers {
            let lp = format!("{p}.layers.{i}");
            layers.push(EncoderLayer {
                q_w: weights.take(&format!("{lp}.self_attn.q_proj.weight"))?,
                q_b: weights.take(&format!("{lp}.self_attn.q_proj.bias"))?,
                k_w: weights.take(&format!("{lp}.self_attn.k_proj.weight"))?,
                k_b: weights.take(&format!("{lp}.self_attn.k_proj.bias"))?,
                v_w: weights.take(&format!("{lp}.self_attn.v_proj.weight"))?,
                v_b: weights.take(&format!("{lp}.self_attn.v_proj.bias"))?,
                out_w: weights.take(&format!("{lp}.self_attn.out_proj.weight"))?,
                out_b: weights.take(&format!("{lp}.self_attn.out_proj.bias"))?,
                attn_ln_w: weights.take(&format!("{lp}.self_attn_layer_norm.weight"))?,
                attn_ln_b: weights.take(&format!("{lp}.self_attn_layer_norm.bias"))?,
                fc1_w: weights.take(&format!("{lp}.fc1.weight"))?,
                fc1_b: weights.take(&format!("{lp}.fc1.bias"))?,
                fc2_w: weights.take(&format!("{lp}.fc2.weight"))?,
                fc2_b: weights.take(&format!("{lp}.fc2.bias"))?,
                final_ln_w: weights.take(&format!("{lp}.final_layer_norm.weight"))?,
                final_ln_b: weights.take(&format!("{lp}.final_layer_norm.bias"))?,
            });
        }

        let d_model = a.d_model;
        let n_heads = a.encoder_attention_heads;
        Ok(AudioEncoder {
            conv1_w: conv(weights.take(&format!("{p}.conv2d1.weight"))?)?,
            conv1_b: weights.take(&format!("{p}.conv2d1.bias"))?,
            conv2_w: conv(weights.take(&format!("{p}.conv2d2.weight"))?)?,
            conv2_b: weights.take(&format!("{p}.conv2d2.bias"))?,
            conv3_w: conv(weights.take(&format!("{p}.conv2d3.weight"))?)?,
            conv3_b: weights.take(&format!("{p}.conv2d3.bias"))?,
            conv_out_w: weights.take(&format!("{p}.conv_out.weight"))?,
            layers,
            ln_post_w: weights.take(&format!("{p}.ln_post.weight"))?,
            ln_post_b: weights.take(&format!("{p}.ln_post.bias"))?,
            proj1_w: weights.take(&format!("{p}.proj1.weight"))?,
            proj1_b: weights.take(&format!("{p}.proj1.bias"))?,
            proj2_w: weights.take(&format!("{p}.proj2.weight"))?,
            proj2_b: weights.take(&format!("{p}.proj2.bias"))?,
            pos_embedding: sinusoids_position_embedding(a.max_source_positions, d_model)?,
            n_heads,
            head_dim: d_model / n_heads,
            chunk_len: cfg.chunk_len(),
            window_tokens: 13 * (a.n_window_infer / cfg.chunk_len()),
        })
    }

    /// Encode a log-mel spectrogram into `[num_audio_tokens, output_dim]`
    /// LM-space embeddings (bf16).
    pub fn forward(&self, mel: &LogMelSpectrogram) -> Result<Array> {
        let n_mels = mel.n_mels as i32;
        let n_frames = mel.n_frames as i32;
        let chunk_len = self.chunk_len;
        debug_assert_eq!(n_frames % chunk_len, 0);
        let num_chunks = n_frames / chunk_len;

        // [n_mels, frames] -> [chunks, n_mels, chunk_len, 1] (NHWC, C=1)
        let x = Array::from_slice(&mel.data, &[n_mels, n_frames]);
        let x = x.as_dtype(Dtype::Bfloat16)?;
        let x = x
            .reshape(&[n_mels, num_chunks, chunk_len])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[num_chunks, n_mels, chunk_len, 1])?;

        let conv = |x: &Array, w: &Array, b: &Array| -> Result<Array> {
            let y = ops::conv2d(x, w, (2, 2), (1, 1), (1, 1), 1)?;
            gelu(&ops::add(&y, b)?)
        };
        let x = conv(&x, &self.conv1_w, &self.conv1_b)?;
        let x = conv(&x, &self.conv2_w, &self.conv2_b)?;
        let x = conv(&x, &self.conv3_w, &self.conv3_b)?;

        // [N, freq, time, ch] -> [N, time, ch*freq] (channel-major, then freq)
        let (freq_bins, time_steps, channels) = (x.shape()[1], x.shape()[2], x.shape()[3]);
        let x = x.transpose_axes(&[0, 2, 3, 1])?.reshape(&[
            num_chunks,
            time_steps,
            channels * freq_bins,
        ])?;
        let x = linear(&x, &self.conv_out_w, None)?;

        // Add sinusoidal positions (per chunk, positions 0..time_steps).
        let pos = self
            .pos_embedding
            .index((0..time_steps, ..))
            .as_dtype(x.dtype())?;
        let x = ops::add(&x, &pos)?;

        // Select valid (non-padding) post-CNN tokens into a packed sequence.
        let d_model = x.shape()[2];
        let x = x.reshape(&[num_chunks * time_steps, d_model])?;
        let valid_indices = valid_token_indices(
            mel.n_valid_frames as i64,
            chunk_len as i64,
            num_chunks as i64,
            time_steps as i64,
        );
        let idx = Array::from_slice(
            &valid_indices.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            &[valid_indices.len() as i32],
        );
        let mut hidden = take_axis(&x, &idx, 0)?;

        // Block-diagonal attention windows over the packed sequence.
        let total = hidden.shape()[0];
        let windows = window_boundaries(total, self.window_tokens);

        for layer in &self.layers {
            hidden = self.layer_forward(layer, &hidden, &windows)?;
        }

        let hidden = layer_norm(&hidden, &self.ln_post_w, &self.ln_post_b, LN_EPS)?;

        // Multi-modal projector (proj1 -> gelu -> proj2).
        let hidden = linear(&hidden, &self.proj1_w, Some(&self.proj1_b))?;
        let hidden = gelu(&hidden)?;
        let out = linear(&hidden, &self.proj2_w, Some(&self.proj2_b))?;
        out.eval()?;
        Ok(out)
    }

    fn layer_forward(
        &self,
        layer: &EncoderLayer,
        hidden: &Array,
        windows: &[(i32, i32)],
    ) -> Result<Array> {
        let t = hidden.shape()[0];
        let normed = layer_norm(hidden, &layer.attn_ln_w, &layer.attn_ln_b, LN_EPS)?;

        let q = linear(&normed, &layer.q_w, Some(&layer.q_b))?;
        let k = linear(&normed, &layer.k_w, Some(&layer.k_b))?;
        let v = linear(&normed, &layer.v_w, Some(&layer.v_b))?;

        // [T, D] -> [1, heads, T, head_dim]
        let split_heads = |a: &Array| -> Result<Array> {
            Ok(a.reshape(&[1, t, self.n_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = split_heads(&q)?;
        let k = split_heads(&k)?;
        let v = split_heads(&v)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut outputs = Vec::with_capacity(windows.len());
        for &(start, end) in windows {
            let qw = q.index((.., .., start..end, ..));
            let kw = k.index((.., .., start..end, ..));
            let vw = v.index((.., .., start..end, ..));
            let o = scaled_dot_product_attention(&qw, &kw, &vw, scale, None)?;
            outputs.push(o);
        }
        let refs: Vec<&Array> = outputs.iter().collect();
        let attn = ops::concatenate_axis(&refs, 2)?;

        // [1, heads, T, hd] -> [T, D]
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[t, self.n_heads * self.head_dim])?;
        let attn = linear(&attn, &layer.out_w, Some(&layer.out_b))?;
        let hidden = ops::add(hidden, &attn)?;

        let normed = layer_norm(&hidden, &layer.final_ln_w, &layer.final_ln_b, LN_EPS)?;
        let mlp = linear(&normed, &layer.fc1_w, Some(&layer.fc1_b))?;
        let mlp = gelu(&mlp)?;
        let mlp = linear(&mlp, &layer.fc2_w, Some(&layer.fc2_b))?;
        Ok(ops::add(&hidden, &mlp)?)
    }
}

/// Length after one (k=3, s=2, p=1) convolution; zero stays zero.
fn post_conv_len(l: i64) -> i64 {
    if l > 0 {
        (l - 1) / 2 + 1
    } else {
        0
    }
}

/// Row indices (into the `[num_chunks * time_steps]` flattened conv output)
/// of tokens that correspond to real (unpadded) mel frames.
fn valid_token_indices(
    n_valid_frames: i64,
    chunk_len: i64,
    num_chunks: i64,
    time_steps: i64,
) -> Vec<i64> {
    let mut indices = Vec::new();
    for chunk in 0..num_chunks {
        let start = chunk * chunk_len;
        let valid_in_chunk = (n_valid_frames - start).clamp(0, chunk_len);
        let post = post_conv_len(post_conv_len(post_conv_len(valid_in_chunk)));
        for t in 0..post.min(time_steps) {
            indices.push(chunk * time_steps + t);
        }
    }
    indices
}

/// Split `total` packed tokens into attention windows of `window_tokens`
/// (final window keeps the remainder), mirroring `get_audio_cu_seqlens`.
fn window_boundaries(total: i32, window_tokens: i32) -> Vec<(i32, i32)> {
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < total {
        let end = (start + window_tokens).min(total);
        bounds.push((start, end));
        start = end;
    }
    bounds
}

/// Whisper-style sinusoidal position embedding `[length, channels]` (f32).
fn sinusoids_position_embedding(length: i32, channels: i32) -> Result<Array> {
    let half = channels / 2;
    let log_timescale_increment = (10000f64).ln() / (half as f64 - 1.0);
    let mut data = vec![0.0f32; (length * channels) as usize];
    for pos in 0..length {
        for i in 0..half {
            let inv_timescale = (-log_timescale_increment * i as f64).exp();
            let scaled = pos as f64 * inv_timescale;
            data[(pos * channels + i) as usize] = scaled.sin() as f32;
            data[(pos * channels + half + i) as usize] = scaled.cos() as f32;
        }
    }
    Ok(Array::from_slice(&data, &[length, channels]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_indices_full_and_partial_chunks() {
        // 150 valid frames over 2 chunks of 100: 13 + 7 tokens.
        let idx = valid_token_indices(150, 100, 2, 13);
        assert_eq!(idx.len(), 20);
        assert_eq!(idx[13], 13); // second chunk starts at row 13
                                 // Fully padded chunk contributes nothing.
        let idx = valid_token_indices(100, 100, 2, 13);
        assert_eq!(idx.len(), 13);
    }

    #[test]
    fn windows_cover_sequence() {
        assert_eq!(
            window_boundaries(260, 104),
            vec![(0, 104), (104, 208), (208, 260)]
        );
        assert_eq!(window_boundaries(7, 104), vec![(0, 7)]);
    }
}
