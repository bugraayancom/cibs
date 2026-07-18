//! High-level Qwen3-ASR transcription pipeline.

use std::path::Path;
use std::time::Instant;

use crate::audio;
use crate::config::{resolve_language, ModelConfig};
use crate::decoder::Decoder;
use crate::encoder::AudioEncoder;
use crate::error::Result;
use crate::generate::{self, GenerateConfig};
use crate::mel::log_mel_spectrogram;
use crate::tokenizer::{parse_output, AsrTokenizer, ParsedOutput};
use crate::weights::Weights;

/// End-to-end ASR model (encoder + decoder + tokenizer).
pub struct AsrModel {
    pub config: ModelConfig,
    pub tokenizer: AsrTokenizer,
    encoder: AudioEncoder,
    decoder: Decoder,
}

/// Result of a single transcription request.
#[derive(Debug, Clone)]
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
    pub raw: String,
    pub audio_seconds: f64,
    pub elapsed_seconds: f64,
    pub rtf: f64,
    pub num_audio_tokens: usize,
    pub num_generated_tokens: usize,
}

impl AsrModel {
    /// Load config, tokenizer, and all weights from a HuggingFace model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        tracing::info!(path = %model_dir.display(), "loading model config");
        let config = ModelConfig::load(model_dir)?;
        let tokenizer = AsrTokenizer::load(model_dir)?;

        tracing::info!("loading safetensors weights");
        let mut weights = Weights::load(model_dir)?;
        tracing::info!(tensors = weights.len(), "weights mapped");

        let encoder = AudioEncoder::load(&mut weights, &config)?;
        let decoder = Decoder::load(&mut weights, &config)?;

        if !weights.is_empty() {
            tracing::debug!(
                leftover = weights.len(),
                "unused tensors remain in checkpoint (expected for unused keys)"
            );
        }

        Ok(AsrModel {
            config,
            tokenizer,
            encoder,
            decoder,
        })
    }

    /// Transcribe a PCM waveform already resampled to 16 kHz mono f32.
    pub fn transcribe_samples(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<Transcription> {
        let (_, transcription) = self.run(samples, language, &[], max_new_tokens)?;
        Ok(transcription)
    }

    /// One streaming step: re-encode all accumulated audio and continue
    /// generation from `forced_tokens` (the previous step's response tokens
    /// minus a few rollback tokens, per the official streaming recipe).
    ///
    /// Returns the full raw response token sequence (forced + newly
    /// generated) plus the parsed transcription.
    pub fn transcribe_stream_step(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        forced_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<(Vec<u32>, Transcription)> {
        self.run(samples, language, forced_tokens, max_new_tokens)
    }

    fn run(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        forced_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<(Vec<u32>, Transcription)> {
        let t0 = Instant::now();
        let audio_seconds = samples.len() as f64 / audio::SAMPLE_RATE as f64;

        let lang = resolve_language(language)?;
        let (prefix, suffix) = self.tokenizer.build_prompt(lang.as_deref(), None)?;

        let mel = log_mel_spectrogram(
            samples,
            self.config.audio.num_mel_bins as usize,
            self.config.chunk_len() as usize,
        )?;
        tracing::debug!(
            frames = mel.n_frames,
            valid = mel.n_valid_frames,
            "mel spectrogram ready"
        );

        let audio_embeds = self.encoder.forward(&mel)?;
        let num_audio_tokens = audio_embeds.shape()[0] as usize;
        tracing::debug!(num_audio_tokens, "encoder finished");

        let input_embeds = generate::build_input_embeds(
            &self.decoder,
            &prefix,
            &audio_embeds,
            &suffix,
            forced_tokens,
        )?;

        let gen_cfg = GenerateConfig {
            max_new_tokens,
            eos_token_ids: self.config.generation.eos_token_id.clone(),
        };
        let new_tokens =
            generate::generate(&mut self.decoder, &self.tokenizer, &input_embeds, &gen_cfg)?;
        let num_generated_tokens = new_tokens.len();

        let mut tokens = forced_tokens.to_vec();
        tokens.extend_from_slice(&new_tokens);

        // Decode with special tokens kept so `<asr_text>` survives for parsing.
        let raw = self.decode_with_asr_tag(&tokens)?;
        let parsed = parse_output(&raw);

        let elapsed_seconds = t0.elapsed().as_secs_f64();
        let rtf = if audio_seconds > 0.0 {
            elapsed_seconds / audio_seconds
        } else {
            0.0
        };

        let transcription = Transcription {
            text: parsed.text,
            language: parsed.language.or(lang),
            raw,
            audio_seconds,
            elapsed_seconds,
            rtf,
            num_audio_tokens,
            num_generated_tokens,
        };
        Ok((tokens, transcription))
    }

    /// Transcribe an audio file (any format ffmpeg can decode).
    pub fn transcribe_file(
        &mut self,
        path: &Path,
        language: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<Transcription> {
        let samples = audio::decode_file(path)?;
        self.transcribe_samples(&samples, language, max_new_tokens)
    }

    /// Transcribe in-memory audio bytes (HTTP upload path).
    pub fn transcribe_bytes(
        &mut self,
        bytes: &[u8],
        filename_hint: &str,
        language: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<Transcription> {
        let samples = audio::decode_bytes(bytes, filename_hint)?;
        self.transcribe_samples(&samples, language, max_new_tokens)
    }

    fn decode_with_asr_tag(&self, tokens: &[u32]) -> Result<String> {
        // Keep `<asr_text>` (151704) visible; skip other specials via tokenizer.decode.
        // The HF tokenizer's skip_special_tokens drops `<asr_text>`, so decode in
        // segments around that id.
        const ASR_TEXT: u32 = 151704;
        let mut parts = Vec::new();
        let mut start = 0usize;
        for (i, &tid) in tokens.iter().enumerate() {
            if tid == ASR_TEXT {
                if start < i {
                    parts.push(self.tokenizer.decode(&tokens[start..i])?);
                }
                parts.push("<asr_text>".to_string());
                start = i + 1;
            }
        }
        if start < tokens.len() {
            parts.push(self.tokenizer.decode(&tokens[start..])?);
        }
        Ok(parts.concat())
    }
}

/// Convenience re-export for callers that only need the parsed fields.
pub fn parsed_from_transcription(t: &Transcription) -> ParsedOutput {
    ParsedOutput {
        language: t.language.clone(),
        text: t.text.clone(),
    }
}
