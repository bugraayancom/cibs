//! Greedy autoregressive decoding for Qwen3-ASR.

use mlx_rs::{ops, Array};

use crate::decoder::Decoder;
use crate::error::{Error, Result};
use crate::tokenizer::AsrTokenizer;

/// Generation hyperparameters.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 1024,
            eos_token_ids: vec![151643, 151645],
        }
    }
}

/// Prefill `input_embeds` then greedily decode until EOS or `max_new_tokens`.
///
/// Returns the generated token ids (EOS stripped).
pub fn generate(
    decoder: &mut Decoder,
    tokenizer: &AsrTokenizer,
    input_embeds: &Array,
    cfg: &GenerateConfig,
) -> Result<Vec<u32>> {
    let _ = tokenizer;
    generate_from_decoder(decoder, input_embeds, cfg)
}

/// Tokenizer-agnostic greedy generation loop (used by both the ASR pipeline
/// and the standalone translator).
pub fn generate_from_decoder(
    decoder: &mut Decoder,
    input_embeds: &Array,
    cfg: &GenerateConfig,
) -> Result<Vec<u32>> {
    if input_embeds.ndim() != 2 {
        return Err(Error::Generate(format!(
            "expected [seq, hidden] embeddings, got shape {:?}",
            input_embeds.shape()
        )));
    }

    let logits = decoder.prefill(input_embeds)?;
    let mut token = Decoder::argmax_token(&logits)?;
    tracing::debug!(first_token = token, "prefill complete");
    let mut generated = Vec::with_capacity(cfg.max_new_tokens.min(64));

    for step in 0..cfg.max_new_tokens {
        if cfg.eos_token_ids.contains(&token) {
            tracing::debug!(step, token, "hit EOS");
            break;
        }
        generated.push(token);

        let emb = decoder.embed(&[token])?;
        let logits = decoder.decode_step(&emb)?;
        token = Decoder::argmax_token(&logits)?;
        if step < 8 {
            tracing::debug!(step, token, "decode step");
        }
    }

    tracing::debug!(?generated, "generated token ids");
    Ok(generated)
}

/// Build the decoder input embedding sequence:
/// `embed(prefix) ‖ audio_embeds ‖ embed(suffix) ‖ embed(forced)`.
///
/// `forced` carries already-committed response tokens (streaming prefix
/// rollback); pass an empty slice for one-shot transcription.
pub fn build_input_embeds(
    decoder: &Decoder,
    prefix: &[u32],
    audio_embeds: &Array,
    suffix: &[u32],
    forced: &[u32],
) -> Result<Array> {
    let prefix_e = decoder.embed(prefix)?;
    let suffix_e = decoder.embed(suffix)?;
    let audio = audio_embeds.as_dtype(prefix_e.dtype())?;
    if forced.is_empty() {
        return Ok(ops::concatenate_axis(&[&prefix_e, &audio, &suffix_e], 0)?);
    }
    let forced_e = decoder.embed(forced)?;
    Ok(ops::concatenate_axis(
        &[&prefix_e, &audio, &suffix_e, &forced_e],
        0,
    )?)
}
