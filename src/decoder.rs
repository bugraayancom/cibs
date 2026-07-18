//! Qwen3 causal LM decoder with GQA, Q/K RMSNorm, NeoX RoPE, and KV cache.

use mlx_rs::fast::{rms_norm, rope, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{argmax, IndexOp};
use mlx_rs::{nn, ops, Array, Dtype};

use crate::config::ModelConfig;
use crate::error::Result;
use crate::qlinear::{Embedding, Linear, QuantParams};
use crate::weights::Weights;

struct DecoderLayer {
    input_ln: Array,
    post_ln: Array,
    q_w: Linear,
    k_w: Linear,
    v_w: Linear,
    o_w: Linear,
    q_norm: Array,
    k_norm: Array,
    gate_w: Linear,
    up_w: Linear,
    down_w: Linear,
}

struct KvCache {
    k: Array,
    v: Array,
}

/// Qwen3 text decoder used by the ASR thinker.
pub struct Decoder {
    embed_tokens: Embedding,
    lm_head: Linear,
    norm: Array,
    layers: Vec<DecoderLayer>,
    caches: Vec<Option<KvCache>>,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    eps: f32,
    rope_theta: f32,
    /// Current filled length of the KV cache (absolute next position).
    offset: i32,
}

impl Decoder {
    pub fn load(weights: &mut Weights, cfg: &ModelConfig) -> Result<Self> {
        Self::load_with_prefix(weights, &cfg.text, "thinker.", None)
    }

    /// Load a Qwen3 decoder whose tensors live under `prefix` (`"thinker."`
    /// for the ASR checkpoint, `""` for a standalone Qwen3 LM). Pass `quant`
    /// for MLX-quantized checkpoints (4/8-bit).
    pub fn load_with_prefix(
        weights: &mut Weights,
        t: &crate::config::TextConfig,
        prefix: &str,
        quant: Option<QuantParams>,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(t.num_hidden_layers as usize);
        for i in 0..t.num_hidden_layers {
            let lp = format!("{prefix}model.layers.{i}");
            layers.push(DecoderLayer {
                input_ln: weights.take(&format!("{lp}.input_layernorm.weight"))?,
                post_ln: weights.take(&format!("{lp}.post_attention_layernorm.weight"))?,
                q_w: Linear::take(weights, &format!("{lp}.self_attn.q_proj"), quant)?,
                k_w: Linear::take(weights, &format!("{lp}.self_attn.k_proj"), quant)?,
                v_w: Linear::take(weights, &format!("{lp}.self_attn.v_proj"), quant)?,
                o_w: Linear::take(weights, &format!("{lp}.self_attn.o_proj"), quant)?,
                q_norm: weights.take(&format!("{lp}.self_attn.q_norm.weight"))?,
                k_norm: weights.take(&format!("{lp}.self_attn.k_norm.weight"))?,
                gate_w: Linear::take(weights, &format!("{lp}.mlp.gate_proj"), quant)?,
                up_w: Linear::take(weights, &format!("{lp}.mlp.up_proj"), quant)?,
                down_w: Linear::take(weights, &format!("{lp}.mlp.down_proj"), quant)?,
            });
        }

        let embed_tokens =
            Embedding::take(weights, &format!("{prefix}model.embed_tokens"), quant)?;
        // Prefer an explicit lm_head when present; fall back to tied embeddings.
        let lm_head = if weights.contains(&format!("{prefix}lm_head.weight")) {
            Linear::take(weights, &format!("{prefix}lm_head"), quant)?
        } else {
            embed_tokens.as_linear()
        };
        let norm = weights.take(&format!("{prefix}model.norm.weight"))?;

        let n_layers = layers.len();
        Ok(Decoder {
            embed_tokens,
            lm_head,
            norm,
            layers,
            caches: (0..n_layers).map(|_| None).collect(),
            n_heads: t.num_attention_heads,
            n_kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            eps: t.rms_norm_eps,
            rope_theta: t.rope_theta,
            offset: 0,
        })
    }

    /// Reset KV cache between independent requests.
    pub fn reset_cache(&mut self) {
        for c in &mut self.caches {
            *c = None;
        }
        self.offset = 0;
    }

    /// Look up token embeddings: `ids` → `[seq, hidden]`.
    pub fn embed(&self, ids: &[u32]) -> Result<Array> {
        let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
        let idx = Array::from_slice(&ids_i32, &[ids_i32.len() as i32]);
        self.embed_tokens.lookup(&idx)
    }

    /// Prefill the full prompt embedding sequence and return logits for the
    /// last position (`[vocab]`).
    pub fn prefill(&mut self, embeds: &Array) -> Result<Array> {
        self.reset_cache();
        let seq = embeds.shape()[0];
        let hidden = self.forward_layers(embeds, 0)?;
        self.offset = seq;
        let last = hidden.index((seq - 1, ..));
        self.logits(&last)
    }

    /// Decode one new token embedding at the current cache offset.
    /// `embed` shape: `[1, hidden]` or `[hidden]`.
    pub fn decode_step(&mut self, embed: &Array) -> Result<Array> {
        let x = if embed.ndim() == 1 {
            embed.reshape(&[1, -1])?
        } else {
            embed.clone()
        };
        let pos = self.offset;
        let hidden = self.forward_layers(&x, pos)?;
        self.offset = pos + x.shape()[0];
        let last = hidden.index((x.shape()[0] - 1, ..));
        self.logits(&last)
    }

    fn logits(&self, hidden: &Array) -> Result<Array> {
        let h = rms_norm(hidden, &self.norm, self.eps)?;
        let h = h.reshape(&[1, -1])?;
        let out = match &self.lm_head {
            // Compute dense logits in f32 for numerically stable argmax.
            Linear::Dense(w) => {
                let h = h.as_dtype(Dtype::Float32)?;
                let w = w.as_dtype(Dtype::Float32)?;
                ops::matmul(&h, &w.transpose()?)?
            }
            // quantized_matmul runs in the activation dtype (bf16); cast after.
            q => q.apply(&h)?.as_dtype(Dtype::Float32)?,
        };
        Ok(out.reshape(&[-1])?)
    }

    /// Greedy argmax over a `[vocab]` logits vector.
    pub fn argmax_token(logits: &Array) -> Result<u32> {
        let idx = argmax(logits, None)?;
        idx.eval()?;
        Ok(idx.item::<u32>() as u32)
    }

    fn forward_layers(&mut self, x: &Array, start_pos: i32) -> Result<Array> {
        let mut h = x.as_dtype(Dtype::Bfloat16)?;
        let n_layers = self.layers.len();
        for i in 0..n_layers {
            h = self.layer_forward(i, &h, start_pos)?;
            // Bound the lazy graph; eval every few layers during long prefills.
            if i % 4 == 3 {
                h.eval()?;
            }
        }
        h.eval()?;
        Ok(h)
    }

    fn layer_forward(&mut self, layer_idx: usize, h: &Array, start_pos: i32) -> Result<Array> {
        // Clone Array handles (cheap refcount) so we can mutate `self.caches`
        // without overlapping borrows of `self.layers`.
        let layer = &self.layers[layer_idx];
        let input_ln = layer.input_ln.clone();
        let post_ln = layer.post_ln.clone();
        let q_w = layer.q_w.clone();
        let k_w = layer.k_w.clone();
        let v_w = layer.v_w.clone();
        let o_w = layer.o_w.clone();
        let q_norm = layer.q_norm.clone();
        let k_norm = layer.k_norm.clone();
        let gate_w = layer.gate_w.clone();
        let up_w = layer.up_w.clone();
        let down_w = layer.down_w.clone();
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let eps = self.eps;
        let rope_theta = self.rope_theta;

        let seq = h.shape()[0];
        let x = rms_norm(h, &input_ln, eps)?;

        let q = q_w.apply(&x)?;
        let k = k_w.apply(&x)?;
        let v = v_w.apply(&x)?;

        // [seq, n_heads, head_dim] for per-head RMSNorm.
        let q = q.reshape(&[seq, n_heads, head_dim])?;
        let k = k.reshape(&[seq, n_kv_heads, head_dim])?;
        let q = rms_norm(&q, &q_norm, eps)?;
        let k = rms_norm(&k, &k_norm, eps)?;

        // [1, heads, seq, head_dim] for SDPA / RoPE.
        let q = q
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[1, n_heads, seq, head_dim])?;
        let k = k
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[1, n_kv_heads, seq, head_dim])?;
        let v = v
            .reshape(&[seq, n_kv_heads, head_dim])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[1, n_kv_heads, seq, head_dim])?;

        // NeoX / split-half RoPE. In MLX, traditional=false rotates the two
        // halves of the head (matching Qwen3 / mlx-lm); traditional=true
        // rotates consecutive pairs instead.
        let q = rope(&q, head_dim, false, rope_theta, 1.0, start_pos, None)?;
        let k = rope(&k, head_dim, false, rope_theta, 1.0, start_pos, None)?;

        // Append to KV cache.
        let (k_full, v_full) = match &self.caches[layer_idx] {
            Some(cache) => {
                let k_cat = ops::concatenate_axis(&[&cache.k, &k], 2)?;
                let v_cat = ops::concatenate_axis(&[&cache.v, &v], 2)?;
                (k_cat, v_cat)
            }
            None => (k, v),
        };
        self.caches[layer_idx] = Some(KvCache {
            k: k_full.clone(),
            v: v_full.clone(),
        });

        let scale = 1.0 / (head_dim as f32).sqrt();
        // Causal mask only needed when query length > 1 (prefill).
        let mask = if seq > 1 {
            Some(ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let attn = scaled_dot_product_attention(&q, &k_full, &v_full, scale, mask)?;

        // [1, heads, seq, hd] -> [seq, hidden]
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[seq, n_heads * head_dim])?;
        let attn = o_w.apply(&attn)?;
        let h = ops::add(h, &attn)?;

        let x = rms_norm(&h, &post_ln, eps)?;
        let gate = nn::silu(gate_w.apply(&x)?)?;
        let up = up_w.apply(&x)?;
        let mlp = down_w.apply(&ops::multiply(&gate, &up)?)?;
        Ok(ops::add(&h, &mlp)?)
    }
}
