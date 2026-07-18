//! Local translation via a standalone Qwen3 instruct model (e.g.
//! `Qwen/Qwen3-0.6B`), reusing the same decoder implementation as the ASR
//! thinker. Fully offline; no external runtime.

use std::path::Path;
use std::time::Instant;

use tokenizers::Tokenizer;

use crate::config::TextConfig;
use crate::decoder::Decoder;
use crate::error::{Error, Result};
use crate::generate::{self, GenerateConfig};
use crate::qlinear::QuantParams;
use crate::weights::Weights;

/// Qwen3 chat-model wrapper specialized for translation prompts.
pub struct Translator {
    decoder: Decoder,
    tokenizer: Tokenizer,
    im_start: u32,
    im_end: u32,
    newline: u32,
    eos: Vec<u32>,
}

impl Translator {
    /// Load a plain Qwen3 LM directory (config.json + tokenizer.json +
    /// model.safetensors, tensors without the ASR `thinker.` prefix).
    pub fn load(model_dir: &Path) -> Result<Self> {
        tracing::info!(path = %model_dir.display(), "loading translator model");

        let cfg_path = model_dir.join("config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| Error::io(&cfg_path, e))?;
        // Plain Qwen3 config keeps the text fields at the top level.
        let text_cfg: TextConfig = serde_json::from_str(&cfg_text)?;

        // MLX-quantized checkpoints (mlx-community/...-4bit) carry a
        // top-level `quantization` object.
        let quant = serde_json::from_str::<serde_json::Value>(&cfg_text)?
            .get("quantization")
            .map(|q| QuantParams {
                group_size: q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
                bits: q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32,
            });
        if let Some(q) = quant {
            tracing::info!(bits = q.bits, group_size = q.group_size, "quantized model");
        }

        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path)?;

        let token_id = |name: &str| -> Result<u32> {
            tokenizer
                .token_to_id(name)
                .ok_or_else(|| Error::Tokenizer(format!("token {name:?} not in vocabulary")))
        };
        let im_start = token_id("<|im_start|>")?;
        let im_end = token_id("<|im_end|>")?;

        let newline = {
            let enc = tokenizer
                .encode("\n", false)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
            match enc.get_ids() {
                [id] => *id,
                other => {
                    return Err(Error::Tokenizer(format!(
                        "expected single newline token, got {other:?}"
                    )))
                }
            }
        };

        let mut weights = Weights::load(model_dir)?;
        let decoder = Decoder::load_with_prefix(&mut weights, &text_cfg, "", quant)?;
        tracing::info!("translator ready");

        Ok(Translator {
            decoder,
            tokenizer,
            im_start,
            im_end,
            newline,
            eos: vec![151643, im_end],
        })
    }

    /// Translate `text` into `target_language` (full name, e.g. "Turkish").
    pub fn translate(&mut self, text: &str, target_language: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }
        let t0 = Instant::now();

        let system = "You are a professional translator. Output ONLY the translation, \
             with no explanations, notes, or quotes."
            .to_string();
        let user = format!(
            "Translate the following text into {target_language}, preserving meaning \
             and tone:\n\n{text}"
        );

        let mut ids = Vec::with_capacity(256);
        let push_text = |ids: &mut Vec<u32>, s: &str| -> Result<()> {
            let enc = self
                .tokenizer
                .encode(s, false)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
            ids.extend_from_slice(enc.get_ids());
            Ok(())
        };

        ids.push(self.im_start);
        push_text(&mut ids, &format!("system\n{system}"))?;
        ids.push(self.im_end);
        ids.push(self.newline);
        ids.push(self.im_start);
        push_text(&mut ids, &format!("user\n{user}"))?;
        ids.push(self.im_end);
        ids.push(self.newline);
        ids.push(self.im_start);
        // Empty <think> block disables Qwen3's thinking mode.
        push_text(&mut ids, "assistant\n<think>\n\n</think>\n\n")?;

        let input_embeds = self.decoder.embed(&ids)?;
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1024,
            eos_token_ids: self.eos.clone(),
        };
        let tokens = generate::generate_from_decoder(&mut self.decoder, &input_embeds, &gen_cfg)?;

        let mut out = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // Defensive: strip any residual think block.
        if let Some(idx) = out.find("</think>") {
            out = out[idx + "</think>".len()..].to_string();
        }
        let out = out.trim().to_string();

        tracing::debug!(
            elapsed_s = format!("{:.2}", t0.elapsed().as_secs_f64()),
            tokens = tokens.len(),
            "translation done"
        );
        Ok(out)
    }
}
