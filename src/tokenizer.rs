use std::path::Path;

use serde::Deserialize;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
use tokenizers::pre_tokenizers::sequence::Sequence as PreSequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

use crate::error::{Error, Result};

/// GPT-2 style split regex used by the Qwen2 tokenizer.
const QWEN2_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug, Deserialize)]
struct AddedTokenSpec {
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: std::collections::HashMap<String, AddedTokenSpec>,
}

/// Wrapper around the HF `tokenizers` BPE tokenizer for Qwen3-ASR.
///
/// The model repo ships `vocab.json` + `merges.txt` (no `tokenizer.json`),
/// so the full pipeline (NFC normalizer, Qwen2 split regex, byte-level BPE)
/// is assembled here. A `tokenizer.json`, if present, takes precedence.
pub struct AsrTokenizer {
    inner: Tokenizer,
    pub im_start: u32,
    pub im_end: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub audio_pad: u32,
    pub asr_text: u32,
    pub newline: u32,
}

impl AsrTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_json = model_dir.join("tokenizer.json");
        let inner = if tokenizer_json.is_file() {
            Tokenizer::from_file(&tokenizer_json)?
        } else {
            Self::build_from_vocab(model_dir)?
        };

        let token_id = |name: &str| -> Result<u32> {
            inner
                .token_to_id(name)
                .ok_or_else(|| Error::Tokenizer(format!("token {name:?} not in vocabulary")))
        };

        let im_start = token_id("<|im_start|>")?;
        let im_end = token_id("<|im_end|>")?;
        let audio_start = token_id("<|audio_start|>")?;
        let audio_end = token_id("<|audio_end|>")?;
        let audio_pad = token_id("<|audio_pad|>")?;
        let asr_text = token_id("<asr_text>")?;

        let newline = {
            let enc = inner
                .encode("\n", false)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
            match enc.get_ids() {
                [id] => *id,
                other => {
                    return Err(Error::Tokenizer(format!(
                        "expected single token for newline, got {other:?}"
                    )))
                }
            }
        };

        Ok(AsrTokenizer {
            inner,
            im_start,
            im_end,
            audio_start,
            audio_end,
            audio_pad,
            asr_text,
            newline,
        })
    }

    fn build_from_vocab(model_dir: &Path) -> Result<Tokenizer> {
        let vocab = model_dir.join("vocab.json");
        let merges = model_dir.join("merges.txt");
        if !vocab.is_file() || !merges.is_file() {
            return Err(Error::Tokenizer(format!(
                "neither tokenizer.json nor vocab.json+merges.txt found in {}",
                model_dir.display()
            )));
        }

        let bpe = BPE::from_file(
            vocab.to_str().unwrap_or_default(),
            merges.to_str().unwrap_or_default(),
        )
        .byte_fallback(false)
        .build()
        .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_normalizer(Some(NFC));

        let split = Split::new(
            SplitPattern::Regex(QWEN2_SPLIT_REGEX.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let byte_level = ByteLevelPre::new(false, false, false);
        tokenizer.with_pre_tokenizer(Some(PreSequence::new(vec![
            split.into(),
            byte_level.into(),
        ])));
        tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, false)));

        // Register added tokens (audio/control tokens) from tokenizer_config.json,
        // in ascending id order so ids line up with the checkpoint.
        let cfg_path = model_dir.join("tokenizer_config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| Error::io(&cfg_path, e))?;
        let cfg: TokenizerConfig = serde_json::from_str(&cfg_text)?;

        let mut added: Vec<(u32, AddedTokenSpec)> = cfg
            .added_tokens_decoder
            .into_iter()
            .map(|(id, spec)| {
                id.parse::<u32>()
                    .map(|id| (id, spec))
                    .map_err(|e| Error::Tokenizer(format!("bad added token id {id:?}: {e}")))
            })
            .collect::<Result<_>>()?;
        added.sort_by_key(|(id, _)| *id);

        for (expect_id, spec) in added {
            let token = AddedToken::from(spec.content.clone(), spec.special);
            tokenizer.add_tokens(&[token]);
            let got = tokenizer.token_to_id(&spec.content);
            if got != Some(expect_id) {
                return Err(Error::Tokenizer(format!(
                    "added token {:?} mapped to id {:?}, expected {}",
                    spec.content, got, expect_id
                )));
            }
        }

        Ok(tokenizer)
    }

    /// Encode plain text (no special-token handling).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token ids, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Build the ASR prompt as (prefix, suffix) token id sequences; the
    /// encoder's audio embeddings are inserted between the two parts
    /// (in place of the `<|audio_pad|>` placeholders of the chat template).
    ///
    /// Layout (matches the model's chat template):
    /// `<|im_start|>system\n{context?}<|im_end|>\n<|im_start|>user\n`
    /// `<|audio_start|>{AUDIO}<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
    ///
    /// When `language` is set, the official forced-language recipe appends
    /// `language {Lang}<asr_text>` right after the generation prompt so the
    /// model must continue with a transcript in that language (instead of
    /// emitting its own auto-detected `language ...<asr_text>` header).
    pub fn build_prompt(
        &self,
        language: Option<&str>,
        context: Option<&str>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let mut prefix = Vec::with_capacity(32);

        // system
        prefix.push(self.im_start);
        prefix.extend(self.encode("system")?);
        prefix.push(self.newline);
        if let Some(ctx) = context {
            if !ctx.is_empty() {
                prefix.extend(self.encode(ctx)?);
            }
        }
        prefix.push(self.im_end);
        prefix.push(self.newline);

        // user (audio)
        prefix.push(self.im_start);
        prefix.extend(self.encode("user")?);
        prefix.push(self.newline);
        prefix.push(self.audio_start);

        // suffix: after the audio embeddings
        let mut suffix = Vec::with_capacity(8);
        suffix.push(self.audio_end);
        suffix.push(self.im_end);
        suffix.push(self.newline);
        suffix.push(self.im_start);
        suffix.extend(self.encode("assistant")?);
        suffix.push(self.newline);
        if let Some(lang) = language {
            suffix.extend(self.encode(&format!("language {lang}"))?);
            suffix.push(self.asr_text);
        }

        Ok((prefix, suffix))
    }
}

/// Parsed model output: `language {Lang}<asr_text>{transcript}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedOutput {
    pub language: Option<String>,
    pub text: String,
}

/// Split the raw decoded string into detected language and transcript.
pub fn parse_output(raw: &str) -> ParsedOutput {
    let raw = raw.trim();
    let Some((prefix, text)) = raw.split_once("<asr_text>") else {
        return ParsedOutput {
            language: None,
            text: raw.to_string(),
        };
    };

    let prefix = prefix.trim();
    let language = if prefix.eq_ignore_ascii_case("language none") {
        None
    } else {
        prefix
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .map(|line| {
                line.strip_prefix("language ")
                    .or_else(|| line.strip_prefix("Language "))
                    .unwrap_or(line)
                    .trim()
                    .to_string()
            })
            .filter(|s| !s.is_empty())
    };

    ParsedOutput {
        language,
        text: text.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_language_and_text() {
        let p = parse_output("language English<asr_text>Hello world.");
        assert_eq!(p.language.as_deref(), Some("English"));
        assert_eq!(p.text, "Hello world.");
    }

    #[test]
    fn parses_missing_tag() {
        let p = parse_output("just text");
        assert_eq!(p.language, None);
        assert_eq!(p.text, "just text");
    }

    #[test]
    fn parses_empty_audio() {
        let p = parse_output("language None<asr_text>");
        assert_eq!(p.language, None);
        assert_eq!(p.text, "");
    }

    #[test]
    fn prompt_matches_reference_ids() {
        let dir =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Qwen3-ASR-0.6B");
        if !dir.join("vocab.json").is_file() {
            eprintln!("skip: model dir missing");
            return;
        }
        let tok = AsrTokenizer::load(&dir).unwrap();
        let (prefix, suffix) = tok.build_prompt(None, None).unwrap();
        assert_eq!(
            prefix,
            vec![151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
        );
        assert_eq!(suffix, vec![151670, 151645, 198, 151644, 77091, 198]);
    }
}
