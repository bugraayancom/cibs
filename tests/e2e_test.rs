//! End-to-end transcription against a local Qwen3-ASR-0.6B checkpoint.
//!
//! Ignored by default (needs ~2 GB weights + Metal). Run with:
//!   cargo test --release --features mlx --test e2e_test -- --ignored --nocapture
//!
//! Note: load the model only once per process — MLX/Metal is not happy when
//! two full checkpoints are constructed back-to-back in the same address space.

#![cfg(feature = "mlx")]

use std::path::PathBuf;

use qwen3_asr::asr::AsrModel;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/Qwen3-ASR-0.6B")
}

fn sample(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("samples")
        .join(name)
}

#[test]
#[ignore]
fn transcribes_english_and_chinese() {
    let dir = model_dir();
    assert!(
        dir.join("model.safetensors").is_file(),
        "missing model at {}",
        dir.display()
    );

    let mut model = AsrModel::load(&dir).expect("load");

    let jfk = sample("jfk16.wav");
    assert!(jfk.is_file(), "missing {}", jfk.display());
    let en = model
        .transcribe_file(&jfk, None, 256)
        .expect("transcribe en");
    assert!(
        en.text.to_lowercase().contains("ask not what your country"),
        "unexpected EN transcript: {:?}",
        en.text
    );
    assert!(en.rtf < 1.0, "RTF too high: {}", en.rtf);

    let zh = sample("zh.wav");
    if zh.is_file() {
        let out = model
            .transcribe_file(&zh, Some("zh"), 128)
            .expect("transcribe zh");
        assert!(
            out.text.contains('你') || out.text.contains("语音"),
            "unexpected ZH transcript: {:?}",
            out.text
        );
    }
}

/// Simulates live mode: feed growing slices of the audio in 2 s steps with
/// prefix rollback, as the `--live` CLI loop does.
#[test]
#[ignore]
fn streaming_simulation_matches_offline() {
    const UNFIXED_CHUNK_NUM: usize = 2;
    const UNFIXED_TOKEN_NUM: usize = 5;

    let dir = model_dir();
    let audio = sample("jfk16.wav");
    assert!(audio.is_file(), "missing {}", audio.display());

    let samples = qwen3_asr::audio::decode_file(&audio).expect("decode");
    let mut model = AsrModel::load(&dir).expect("load");

    let step = 2 * qwen3_asr::audio::SAMPLE_RATE as usize;
    let mut prev_tokens: Vec<u32> = Vec::new();
    let mut step_count = 0usize;

    let mut end = step;
    while end <= samples.len() {
        let forced: Vec<u32> = if step_count < UNFIXED_CHUNK_NUM {
            Vec::new()
        } else {
            let keep = prev_tokens.len().saturating_sub(UNFIXED_TOKEN_NUM);
            prev_tokens[..keep].to_vec()
        };
        let (tokens, tr) = model
            .transcribe_stream_step(&samples[..end], None, &forced, 256)
            .expect("stream step");
        prev_tokens = tokens;
        step_count += 1;
        eprintln!("step {step_count}: {:?}", tr.text);
        end += step;
    }
    // Final pass over the full audio.
    let forced: Vec<u32> = {
        let keep = prev_tokens.len().saturating_sub(UNFIXED_TOKEN_NUM);
        prev_tokens[..keep].to_vec()
    };
    let (_, tr) = model
        .transcribe_stream_step(&samples, None, &forced, 256)
        .expect("final step");
    let last_text = tr.text;
    eprintln!("final: {last_text:?}");

    assert!(
        last_text
            .to_lowercase()
            .contains("ask not what your country"),
        "unexpected streaming transcript: {last_text:?}"
    );
}
