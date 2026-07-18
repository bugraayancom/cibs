use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Audio encoder section of `config.json` (`thinker_config.audio_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub d_model: i32,
    pub downsample_hidden_size: i32,
    pub encoder_attention_heads: i32,
    pub encoder_ffn_dim: i32,
    pub encoder_layers: i32,
    pub max_source_positions: i32,
    pub n_window: i32,
    pub n_window_infer: i32,
    pub num_mel_bins: i32,
    pub output_dim: i32,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: i32,
}

fn default_conv_chunksize() -> i32 {
    500
}

/// Text decoder section of `config.json` (`thinker_config.text_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub head_dim: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub max_position_embeddings: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ThinkerConfig {
    audio_config: AudioConfig,
    text_config: TextConfig,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    thinker_config: ThinkerConfig,
    #[serde(default)]
    support_languages: Vec<String>,
}

/// `generation_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            eos_token_id: vec![151643, 151645],
            pad_token_id: 151643,
        }
    }
}

/// Fully resolved model configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub audio: AudioConfig,
    pub text: TextConfig,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
    pub support_languages: Vec<String>,
    pub generation: GenerationConfig,
}

impl ModelConfig {
    /// Load `config.json` (+ optional `generation_config.json`) from a model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| Error::io(&config_path, e))?;
        let raw: RawConfig = serde_json::from_str(&text)?;

        let generation = match std::fs::read_to_string(model_dir.join("generation_config.json")) {
            Ok(s) => serde_json::from_str(&s)?,
            Err(_) => GenerationConfig::default(),
        };

        let thinker = raw.thinker_config;
        let cfg = ModelConfig {
            audio: thinker.audio_config,
            text: thinker.text_config,
            audio_start_token_id: thinker.audio_start_token_id,
            audio_end_token_id: thinker.audio_end_token_id,
            audio_token_id: thinker.audio_token_id,
            support_languages: raw.support_languages,
            generation,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.audio.n_window <= 0 || self.audio.n_window_infer % (self.audio.n_window * 2) != 0 {
            return Err(Error::Config(format!(
                "n_window_infer ({}) must be a positive multiple of n_window*2 ({})",
                self.audio.n_window_infer,
                self.audio.n_window * 2
            )));
        }
        if self.text.num_attention_heads % self.text.num_key_value_heads != 0 {
            return Err(Error::Config(
                "num_attention_heads must be divisible by num_key_value_heads".into(),
            ));
        }
        Ok(())
    }

    /// Mel-frame chunk length consumed by the conv frontend (n_window * 2).
    pub fn chunk_len(&self) -> i32 {
        self.audio.n_window * 2
    }

    /// Number of LM tokens produced for `mel_frames` valid mel frames.
    ///
    /// Mirrors `_get_feat_extract_output_lengths` in the HF implementation:
    /// full chunks contribute 13 tokens each, the final partial chunk goes
    /// through three stride-2 convolutions.
    pub fn audio_token_count(&self, mel_frames: i64) -> i64 {
        let chunk_len = self.chunk_len() as i64;
        let leave = mel_frames % chunk_len;
        // Python-style floor division of (possibly negative) numerator.
        let conv = |n: i64| -> i64 {
            if n <= 0 {
                0
            } else {
                (n - 1) / 2 + 1
            }
        };
        conv(conv(conv(leave))) + (mel_frames / chunk_len) * 13
    }
}

/// Canonical language names accepted by the model, keyed by ISO-style code.
pub const LANGUAGE_CODE_TO_NAME: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("fil", "Filipino"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
];

/// Resolve a language code (`"en"`) or full name (`"English"`) to the
/// canonical full name used in the system prompt. `None` means auto-detect.
pub fn resolve_language(language: Option<&str>) -> Result<Option<String>> {
    let Some(lang) = language else {
        return Ok(None);
    };
    let lower = lang.to_lowercase();
    for (code, name) in LANGUAGE_CODE_TO_NAME {
        if lower == *code || lower == name.to_lowercase() {
            return Ok(Some((*name).to_string()));
        }
    }
    Err(Error::Config(format!(
        "unsupported language: {lang:?} (use a code like 'en' or a name like 'English')"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_token_count_matches_reference() {
        let cfg = test_cfg();
        // Full chunks only: L=500 -> 5*13
        assert_eq!(cfg.audio_token_count(500), 65);
        // 1 second of audio = 100 frames = 1 chunk = 13 tokens
        assert_eq!(cfg.audio_token_count(100), 13);
        // Partial chunk: L=50 -> conv3(conv(conv(50))) = conv(conv(25))=conv(13)=7
        assert_eq!(cfg.audio_token_count(50), 7);
        assert_eq!(cfg.audio_token_count(150), 13 + 7);
    }

    #[test]
    fn language_resolution() {
        assert_eq!(
            resolve_language(Some("tr")).unwrap(),
            Some("Turkish".into())
        );
        assert_eq!(
            resolve_language(Some("english")).unwrap(),
            Some("English".into())
        );
        assert_eq!(resolve_language(None).unwrap(), None);
        assert!(resolve_language(Some("klingon")).is_err());
    }

    fn test_cfg() -> ModelConfig {
        ModelConfig {
            audio: AudioConfig {
                d_model: 896,
                downsample_hidden_size: 480,
                encoder_attention_heads: 14,
                encoder_ffn_dim: 3584,
                encoder_layers: 18,
                max_source_positions: 1500,
                n_window: 50,
                n_window_infer: 800,
                num_mel_bins: 128,
                output_dim: 1024,
                conv_chunksize: 500,
            },
            text: TextConfig {
                head_dim: 128,
                hidden_size: 1024,
                intermediate_size: 3072,
                max_position_embeddings: 65536,
                num_attention_heads: 16,
                num_hidden_layers: 28,
                num_key_value_heads: 8,
                rms_norm_eps: 1e-6,
                rope_theta: 1e6,
                vocab_size: 151936,
                tie_word_embeddings: true,
            },
            audio_start_token_id: 151669,
            audio_end_token_id: 151670,
            audio_token_id: 151676,
            support_languages: vec![],
            generation: GenerationConfig::default(),
        }
    }
}
